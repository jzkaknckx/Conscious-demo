import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

_EPS = 1e-12


def make_gaussian_kernel(kernel_size: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * (sigma**2)))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


class OneShotDerivativeExtractor(nn.Module):
    """
    Compute smoothed image, first derivatives (dx,dy) and second derivatives (dxx,dyy,dxy)
    exactly once per forward call. Later multi-scale processing will pool these maps
    rather than recomputing convolutional derivatives for each scale.

    Notes:
      - base_sigma: smoothing applied before derivatives (defaults to 1.0)
      - sobel kernels are defined once as buffers

    Outputs (all tensors are single-channel):
      dx, dy, dxx, dyy, dxy  with shapes [B,1,H,W]
    """
    def __init__(self, base_sigma: float = 1.0):
        super().__init__()
        self.base_sigma = base_sigma
        sobel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        sobel_y = sobel_x.t()
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _to_luminance(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        r, g, b = 0.2989, 0.5870, 0.1140
        return (r * x[:, 0:1] + g * x[:, 1:2] + b * x[:, 2:3])

    def forward(self, x: torch.Tensor) -> dict:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        x_lum = self._to_luminance(x)

        # base Gaussian smoothing
        sigma = max(0.5, float(self.base_sigma))
        kernel_size = max(3, int(2 * math.ceil(3 * sigma) + 1))
        gauss = make_gaussian_kernel(kernel_size, sigma, device=device, dtype=dtype)
        padding = kernel_size // 2
        x_smooth = F.conv2d(x_lum, gauss, padding=padding, groups=1)

        sobel_x = self.sobel_x.to(device=device, dtype=dtype)
        sobel_y = self.sobel_y.to(device=device, dtype=dtype)
        dx = F.conv2d(x_smooth, sobel_x, padding=1, groups=1)
        dy = F.conv2d(x_smooth, sobel_y, padding=1, groups=1)
        dxx = F.conv2d(dx, sobel_x, padding=1, groups=1)
        dyy = F.conv2d(dy, sobel_y, padding=1, groups=1)
        dxy = F.conv2d(dx, sobel_y, padding=1, groups=1)

        return {
            'x_lum': x_lum,
            'x_smooth': x_smooth,
            'dx': dx,
            'dy': dy,
            'dxx': dxx,
            'dyy': dyy,
            'dxy': dxy
        }


# ---------- feature computations given pooled derivative maps ----------

def curvature_from_prepooled(dxx, dyy, dxy, eps=1e-8):
    curv = torch.sqrt(dxx * dxx + 2.0 * (dxy * dxy) + dyy * dyy + eps)
    B = curv.shape[0]
    flat = curv.view(B, -1)
    maxv = torch.clamp(flat.max(dim=1)[0], min=1e-6).view(B, 1, 1, 1)
    return curv / maxv


def aspect_from_prepooled_J(Jxx, Jyy, Jxy):
    trace = Jxx + Jyy
    disc = torch.clamp((trace * trace) * 0.25 - (Jxx * Jyy - Jxy * Jxy), min=0.0)
    # sqrt_disc = torch.sqrt(disc + _EPS)
    sqrt_disc = torch.sqrt(disc)
    lambda1 = 0.5 * trace + sqrt_disc
    lambda2 = 0.5 * trace - sqrt_disc
    ratio = torch.sqrt((lambda1 + _EPS) / (lambda2 + _EPS))
    theta = 0.5 * torch.atan2(2.0 * Jxy, (Jxx - Jyy) + _EPS)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    verticalness = torch.abs(sin_t) - torch.abs(cos_t)
    sign = torch.sign(verticalness)
    val = sign * torch.log(ratio + 1e-6)
    B = val.shape[0]
    maxv = val.view(B, -1).abs().max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
    return val / maxv


def orientation_from_prepooled_J(Jxx, Jyy, Jxy):
    theta = 0.5 * torch.atan2(2.0 * Jxy, (Jxx - Jyy) + _EPS)
    return theta / (math.pi / 2.0)


# ---------- unified bank that computes derivatives once and returns 3 banks ----------
class MultiScaleFeatureBank(nn.Module):
    """
    Single class that computes dx/dy/dxx/dyy/dxy once and then produces
    multi-scale Curvature / Aspect / Orientation banks without recomputing derivatives.

    forward(x) -> returns tuple of three tensors:
      curvature_bank: [B, S, H, W]
      aspect_bank:    [B, S, H, W]
      orient_bank:    [B, S, H, W]

    Behaviour:
      - For each scale sigma in `scales` we compute a pooling factor `pf = max(1, ceil(sigma))`.
      - We produce pre-pooled derivatives by average-pooling dx,dy,dxx,... or pooling dx*dx, dx*dy for structure tensor.
      - We compute features at pooled resolution and then upsample to original H,W.

    This approach reuses the heavy conv ops once and trades a tiny amount of accuracy for large speedups.
    """
    def __init__(self, scales: List[float], base_sigma: float = 1.0):
        super().__init__()
        assert len(scales) > 0
        self.scales = scales
        self.extractor = OneShotDerivativeExtractor(base_sigma=base_sigma)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        derivs = self.extractor(x)
        dx = derivs['dx']
        dy = derivs['dy']
        dxx = derivs['dxx']
        dyy = derivs['dyy']
        dxy = derivs['dxy']

        curvature_outs = []
        aspect_outs = []
        orient_outs = []

        for s in self.scales:
            pf = max(1, int(math.ceil(s)))
            if pf > 1:
                # pool derivatives (note: pool of squared quantities for structure tensor)
                dx_p = F.avg_pool2d(dx, kernel_size=pf, stride=pf)
                dy_p = F.avg_pool2d(dy, kernel_size=pf, stride=pf)
                dxx_p = F.avg_pool2d(dxx, kernel_size=pf, stride=pf)
                dyy_p = F.avg_pool2d(dyy, kernel_size=pf, stride=pf)
                dxy_p = F.avg_pool2d(dxy, kernel_size=pf, stride=pf)
                Jxx_p = F.avg_pool2d(dx * dx, kernel_size=pf, stride=pf)
                Jyy_p = F.avg_pool2d(dy * dy, kernel_size=pf, stride=pf)
                Jxy_p = F.avg_pool2d(dx * dy, kernel_size=pf, stride=pf)
            else:
                dx_p, dy_p, dxx_p, dyy_p, dxy_p = dx, dy, dxx, dyy, dxy
                Jxx_p = dx * dx
                Jyy_p = dy * dy
                Jxy_p = dx * dy

            # compute features from pooled derivatives
            curv = curvature_from_prepooled(dxx_p, dyy_p, dxy_p)  # [B,1,h_p,w_p]
            asp = aspect_from_prepooled_J(Jxx_p, Jyy_p, Jxy_p)
            ori = orientation_from_prepooled_J(Jxx_p, Jyy_p, Jxy_p)

            # upsample to original size
            if curv.shape[-2:] != (H, W):
                curv_up = F.interpolate(curv, size=(H, W), mode='bilinear', align_corners=False)
                asp_up = F.interpolate(asp, size=(H, W), mode='bilinear', align_corners=False)
                ori_up = F.interpolate(ori, size=(H, W), mode='bilinear', align_corners=False)
            else:
                curv_up, asp_up, ori_up = curv, asp, ori

            curvature_outs.append(curv_up)
            aspect_outs.append(asp_up)
            orient_outs.append(ori_up)

        # concat along scale dimension -> [B, S, H, W]
        curvature_bank = torch.cat(curvature_outs, dim=1)
        aspect_bank = torch.cat(aspect_outs, dim=1)
        orient_bank = torch.cat(orient_outs, dim=1)

        return curvature_bank, aspect_bank, orient_bank


# ----------------------------
# Keep your aggregators (unchanged semantics) but minor fixes for robustness
# ----------------------------
class ScaleAggregator(nn.Module):
    def __init__(self, in_scales: int, out_k: int=1, mode: str='attn'):
        super().__init__()
        assert mode in ('max', 'attn', 'conv')
        self.mode = mode
        self.in_scales = in_scales
        if mode == 'attn':
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
        B, S, H, W = x.shape
        if self.mode == 'max':
            out = torch.max(x, dim=1, keepdim=False)[0]
            return out.unsqueeze(1).repeat(1, self.out_k, 1, 1)
        elif self.mode == 'conv':
            return self.conv1x1(x)
        else:
            logits = self.attn_net(x)  # [B, S*out_k, H, W]
            logits = logits.view(B, self.out_k, S, H, W)
            att = torch.softmax(logits, dim=2)
            x_exp = x.unsqueeze(1).expand(B, self.out_k, S, H, W)
            weighted = (att * x_exp).sum(dim=2)
            return weighted

class AspectAggregator(nn.Module):
    def __init__(self, in_scales: int, out_k: int=2, mode: str='attn'):
        super().__init__()
        assert mode in ('max', 'attn', 'conv')
        self.mode = mode
        self.in_scales = in_scales
        if mode == 'attn':
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
        B, S, H, W = x.shape
        if self.mode == 'max':
            out_max = torch.max(x, dim=1, keepdim=False)[0]
            out_min = torch.min(x, dim=1, keepdim=False)[0]
            out = torch.where(out_max < 0, out_min, out_max)
            return out.unsqueeze(1).repeat(1, self.out_k, 1, 1)
        elif self.mode == 'conv':
            return self.conv1x1(x)
        else:
            logits = self.attn_net(x)
            logits = logits.view(B, self.out_k, S, H, W)
            att = torch.softmax(logits, dim=2)
            x_exp = x.unsqueeze(1).expand(B, self.out_k, S, H, W)
            weighted = (att * x_exp).sum(dim=2)
            return weighted

class OrientationAggregator(nn.Module):
    def __init__(self, in_scales: int, out_k: int=4, mode: str='attn'):
        super().__init__()
        self.in_scales = in_scales
        self.out_k = out_k
        self.mode = mode
        if mode == 'attn':
            self.attn_net = nn.Sequential(
                nn.Conv2d(in_scales*2, max(in_scales*2, 32), kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(max(in_scales*2, 32), in_scales * out_k, kernel_size=1)
            )
        elif mode == 'conv':
            self.conv = nn.Conv2d(in_scales*2, out_k, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H, W = x.shape
        angle = x * (math.pi / 2.0)
        s = torch.sin(angle)
        c = torch.cos(angle)
        comb = torch.cat([s, c], dim=1)  # [B, 2S, H, W]
        if self.mode == 'conv':
            out = self.conv(comb)
            return out
        else:
            logits = self.attn_net(comb)
            logits = logits.view(B, self.out_k, S, H, W)
            att = torch.softmax(logits, dim=2)
            comb_exp = comb.unsqueeze(1).expand(B, self.out_k, 2*S, H, W)
            comb_exp = comb_exp.view(B, self.out_k, S, 2, H, W)
            att = att.unsqueeze(3)
            weighted = (att * comb_exp).sum(dim=2)
            s_agg = weighted[:, :, 0, :, :]
            c_agg = weighted[:, :, 1, :, :]
            theta = torch.atan2(s_agg, c_agg)
            return theta / (math.pi / 2.0)
