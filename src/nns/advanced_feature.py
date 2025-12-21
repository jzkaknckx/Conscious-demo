
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import math
 
# ----------------------------
# === Multi-scale curvature detection layers ===
# ----------------------------


def make_gaussian_kernel(kernel_size: int, sigma: float, channels: int=1):
    """Create a separable Gaussian kernel for conv2d (returns [channels,1,k,k])."""
    ax = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel

class CurvatureFilter(nn.Module):
    """Estimate a curvature-like response for a single receptive field (scale).
    Implementation notes:
      - apply Gaussian smoothing with given sigma (to set receptive field)
      - compute first derivatives (sobel-like) and second derivatives (by differentiating first derivatives)
      - output a positive scalar map proportional to local curvature magnitude

    Output: [B, 1, H, W]
    """
    def __init__(self, sigma: float=1.0, kernel_size: int=None, channels: int=1):
        super().__init__()
        if kernel_size is None:
            # heuristic kernel size from sigma
            kernel_size = max(3, int(2 * math.ceil(3 * sigma) + 1))
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.channels = channels
        # create gaussian kernel as buffer
        kernel = make_gaussian_kernel(self.kernel_size, self.sigma, channels)
        self.register_buffer('gauss', kernel)
        # derivative kernels (Sobel-like)
        sobel_x = torch.tensor([[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]])
        sobel_y = sobel_x.t()
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] (typically C==1 or 3; use luminance if 3)
        B, C, H, W = x.shape
        if C > 1:
            # convert to luminance approx
            x = 0.2989 * x[:,0:1] + 0.5870 * x[:,1:2] + 0.1140 * x[:,2:3]
        # gaussian smoothing
        padding = self.kernel_size // 2
        x_smooth = F.conv2d(x, self.gauss, padding=padding, groups=self.channels)
        # first derivatives
        pad1 = 1
        dx = F.conv2d(x_smooth, self.sobel_x, padding=pad1, groups=self.channels)
        dy = F.conv2d(x_smooth, self.sobel_y, padding=pad1, groups=self.channels)
        # second derivatives (differentiate first derivatives)
        dxx = F.conv2d(dx, self.sobel_x, padding=pad1, groups=self.channels)
        dyy = F.conv2d(dy, self.sobel_y, padding=pad1, groups=self.channels)
        dxy = F.conv2d(dx, self.sobel_y, padding=pad1, groups=self.channels)
        # curvature proxy: Frobenius norm of Hessian
        curv = torch.sqrt(dxx**2 + 2.0 * dxy**2 + dyy**2 + 1e-8)
        # normalize per-batch to stabilize magnitudes (optional)
        curv = curv / (curv.view(B, -1).max(dim=1)[0].view(B,1,1,1) + 1e-6)
        return curv  # [B, 1, H, W]



class MultiScaleCurvatureBank(nn.Module):
    """Bank of curvature filters at multiple receptive field scales.
    Returns: tensor of shape [B, S, H, W] where S == len(scales)
    """
    def __init__(self, scales: List[float]):
        super().__init__()
        self.scales = scales
        self.filters = nn.ModuleList([CurvatureFilter(sigma=s) for s in scales])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        outs = []
        for f in self.filters:
            outs.append(f(x))
        # outs each [B,1,H,W] -> concatenate on channel=1
        return torch.cat(outs, dim=1)  # [B, S, H, W]

class ScaleAggregator(nn.Module):
    """Aggregate multi-scale curvature responses into K curvature channels.
    Two aggregation modes supported:
      - 'max': take per-pixel max across scales (scale-invariant)
      - 'attn': learn per-pixel attention across scales (trainable)
      - 'conv': 1x1 conv that linearly combines scales into K outputs
    Input: [B, S, H, W] -> Output: [B, K, H, W]
    """
    def __init__(self, in_scales: int, out_k: int=1, mode: str='attn'):
        super().__init__()
        assert mode in ('max', 'attn', 'conv')
        self.mode = mode
        self.in_scales = in_scales
        if mode == 'attn':
            # produce attention logits per scale and channel
            # we implement a small network: 1x1 conv to project scales->hidden->scales
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_scales, max(in_scales, 16), kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(max(in_scales, 16), in_scales * out_k, kernel_size=1)
            )
            self.out_k = out_k
        elif mode == 'conv':
            self.conv1x1 = nn.Conv2d(in_scales, out_k, kernel_size=1)
        else:
            self.out_k = out_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H, W]
        B, S, H, W = x.shape
        if self.mode == 'max':
            # reduce across scale dim
            out = torch.max(x, dim=1, keepdim=False)[0]
            # out shape [B, H, W] -> expand to K channels
            return out.unsqueeze(1).repeat(1, self.out_k, 1, 1)
        elif self.mode == 'conv':
            return self.conv1x1(x)
        else:
            logits = self.attn_net(x)  # [B, S*out_k, H, W]
            logits = logits.view(B, self.out_k, S, H, W)  # [B, K, S, H, W]
            # softmax over scales
            att = torch.softmax(logits, dim=2)  # [B, K, S, H, W]
            x_exp = x.unsqueeze(1).expand(B, self.out_k, S, H, W)  # [B,K,S,H,W]
            weighted = (att * x_exp).sum(dim=2)  # [B, K, H, W]
            return weighted

class CurvatureIntegrator(nn.Module):
    """Optional third-layer processing: combine K curvature channels to produce final outputs.
    Modes:
      - 'per_curvature': pass-through (identity)
      - 'fusion': fuse K channels into M output maps via 1x1 conv
      - 'invariant': reduce K channels into a single rotation/scale-robust map (e.g. L2-norm or learned)
    Input: [B, K, H, W]
    Output: [B, M, H, W]
    """
    def __init__(self, in_k: int, out_m: int=1, mode: str='invariant'):
        super().__init__()
        self.mode = mode
        if mode == 'fusion':
            self.conv = nn.Conv2d(in_k, out_m, kernel_size=1)
        elif mode == 'invariant':
            # learned fusion that emphasizes strong responses while being permutation-invariant
            self.fc = nn.Conv2d(in_k, out_m, kernel_size=1)
        else:
            # pass-through
            self.out_m = in_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, H, W]
        if self.mode == 'per_curvature':
            return x
        elif self.mode == 'fusion':
            return self.conv(x)
        else:
            # invariant: compute L2-norm across curvature dims as robust summary, plus learned conv
            l2 = torch.sqrt((x**2).sum(dim=1, keepdim=True) + 1e-8)  # [B,1,H,W]
            learned = self.fc(x)
            # combine
            return torch.cat([l2, learned], dim=1)  # [B, 1+out_m, H, W]

# ----------------------------
# === Convenience wrapper assembling the three groups ===
# ----------------------------
class CurvaturePyramidModule(nn.Module):
    """High-level module that runs:
      - Multi-scale curvature bank -> [B, S, H, W]
      - Scale aggregator -> [B, K, H, W]
      - Curvature integrator -> [B, M, H, W]

    You can configure the number of scales, number of curvature channels and integration mode.
    """
    def __init__(self, scales: List[float], K: int=4, integrator_mode: str='invariant', agg_mode: str='attn'):
        super().__init__()
        self.bank = MultiScaleCurvatureBank(scales)
        self.agg = ScaleAggregator(len(scales), out_k=K, mode=agg_mode)
        # integrator output M channels
        if integrator_mode == 'per_curvature':
            M = K
        elif integrator_mode == 'fusion':
            M = max(1, K//2)
        else:
            M = 1 + max(1, K//4)
        self.int = CurvatureIntegrator(K, out_m=M-1 if M>1 else 0, mode=integrator_mode)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns both intermediate & final: (multi_scale [B,S,H,W], aggregated [B,K,H,W], integrated [B,M,H,W])
        ms = self.bank(x)
        agg = self.agg(ms)
        integ = self.int(agg)
        return ms, agg, integ

# ----------------------------
# === Integration notes (do not duplicate code) ===
# ----------------------------
# To integrate with your RetinaModel pipeline:
#  - call CurvaturePyramidModule with the 'cropped' image output from RetinaModel (or the projection x)
#  - expect shapes:
#       ms:   [B, S, H, W]
#       agg:  [B, K, H, W]
#       integ: [B, M, H, W]
#  - write either agg or integ channels as part of your "高级子模态" that feed into the graph memory layer

# end additions
