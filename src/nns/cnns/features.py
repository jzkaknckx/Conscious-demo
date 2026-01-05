import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from typing import List, Tuple, Optional

_EPS = 1e-12

# ---- shared utility (kept small and deterministic) ----

def make_gaussian_kernel(kernel_size: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * (sigma**2)))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


# ----------------------- advanced_feature (updated) -----------------------
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

    forward(x[, precomputed_derivs]) -> returns tuple of three tensors:
      curvature_bank: [B, S, H, W]
      aspect_bank:    [B, S, H, W]
      orient_bank:    [B, S, H, W]

    If `precomputed_derivs` is provided it should be a dict containing either:
      - 'grad': tensor [...,2] where last dim holds (dx,dy) in shape [B,1,H,W,2]
      - or 'dx','dy' and optionally 'dxx','dyy','dxy' tensors of shape [B,1,H,W]

    This allows reusing derivatives computed by an upstream Retina (to avoid
    duplicate conv operations). If not provided, OneShotDerivativeExtractor is used.
    """
    def __init__(self, scales: List[float], base_sigma: float = 1.0):
        super().__init__()
        assert len(scales) > 0
        self.scales = scales
        self.extractor = OneShotDerivativeExtractor(base_sigma=base_sigma)

    def forward(self, x: torch.Tensor, precomputed_derivs: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,C,H,W]
        B, C, H, W = x.shape

        # If precomputed derivatives are provided (from Retina), reuse them
        if precomputed_derivs is not None:
            device = x.device
            dtype = x.dtype
            # Accept either 'grad' or explicit 'dx'/'dy'
            if 'grad' in precomputed_derivs:
                grad = precomputed_derivs['grad']  # expected [B,1,H,W,2] or [B, H, W, 2]
                # normalize shapes
                if grad.dim() == 5:
                    dx = grad[..., 0].permute(0, 1, 2, 3)
                    dy = grad[..., 1].permute(0, 1, 2, 3)
                elif grad.dim() == 4 and grad.shape[-1] == 2:
                    # [B,H,W,2]
                    dx = grad[..., 0].unsqueeze(1)
                    dy = grad[..., 1].unsqueeze(1)
                else:
                    raise ValueError("Unsupported grad shape: %s" % (grad.shape,))
            else:
                dx = precomputed_derivs.get('dx', None)
                dy = precomputed_derivs.get('dy', None)
                if dx is None or dy is None:
                    raise ValueError("precomputed_derivs must provide 'grad' or both 'dx' and 'dy'")

            # If second derivatives already provided, take them; otherwise compute from dx/dy
            dxx = precomputed_derivs.get('dxx', None)
            dyy = precomputed_derivs.get('dyy', None)
            dxy = precomputed_derivs.get('dxy', None)

            # compute missing second derivatives using sobel kernels from extractor
            sobel_x = self.extractor.sobel_x.to(device=device, dtype=dtype)
            sobel_y = self.extractor.sobel_y.to(device=device, dtype=dtype)
            if dxx is None:
                dxx = F.conv2d(dx, sobel_x, padding=1, groups=1)
            if dyy is None:
                dyy = F.conv2d(dy, sobel_y, padding=1, groups=1)
            if dxy is None:
                dxy = F.conv2d(dx, sobel_y, padding=1, groups=1)

        else:
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


# ----------------------- Retina (updated) -----------------------
class PreprocessLayer(nn.Module):
    '''
        预处理
            切割为cropped_size*cropped_size,以(center_x, center_y)为中心,其余补零

            只能输入B = 1
            未限制(center_x, center_y)输入大小
    '''
    def __init__(self, cropped_size=1024):
        super().__init__()
        self.cropped_size = cropped_size
        self.resize = T.Resize(
            cropped_size,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True
        )

    def forward(self, x, center_x, center_y):
        
        B, C, H, W = x.shape
        device = x.device
        
        if(center_x > W or center_y > H):
            print("Invalid Center")
            
        if(center_x == -1 ): center_x = W//2
        if(center_y == -1 ): center_y = H//2

        # 计算动态裁切窗口
        # crop_h = min(self.cropped_size, H)
        # crop_w = min(self.cropped_size, W)
        crop_h = crop_w = self.cropped_size
        
        # 计算裁切区域坐标
        # start_x = torch.clamp(torch.tensor(center_x - crop_w//2), 0, W - crop_w)
        # start_y = torch.clamp(torch.tensor(center_y - crop_h//2), 0, H - crop_h)
        start_x = center_x - crop_w//2
        start_y = center_y - crop_h//2

        if(center_x < crop_w//2 or W - center_x < crop_w//2 or center_y < crop_h//2 or H - center_y < crop_h//2 or H < crop_h or W < crop_w):
            x = F.pad(x, (crop_w//2, crop_w//2, crop_h//2, crop_h//2))
            start_x += crop_w//2
            start_y += crop_h//2
        
        # 执行裁切
        cropped = []
        
        crop = x[0, :, 
                    int(start_y):int(start_y+crop_h),
                    int(start_x):int(start_x+crop_w)]
        
        # 等比缩放并保持原始比例
        scale = min(self.cropped_size/crop_h, self.cropped_size/crop_w)
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)
        
        resized = T.functional.resize(
            crop, 
            size=[new_h, new_w],
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True
        )
        
        # 边缘填充
        pad_h = (self.cropped_size - new_h)
        pad_w = (self.cropped_size - new_w)
        padded = T.functional.pad(
            resized,
            padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2],
            padding_mode='edge'
        )
        cropped.append(padded)
        
        return torch.cat(cropped, dim=0).unsqueeze(0)


class ProjectionLayer(nn.Module):
    '''
        前向投射
            要求输入正方形

            未缓存
    
    '''
    def __init__(self, output_size=512, center_size=256, tau=0.01):
        super().__init__()
        self.output_size = output_size
        self.center_size = center_size
        self.tau = tau

    def forward(self, x):
        B, C, H_in, W_in = x.shape
        device = x.device
        assert H_in == W_in, "Input must be square"
        cropped_size = H_in

        # if self.tau == -1:
            
        
        # 生成输出网格坐标
        i, j = torch.meshgrid(torch.arange(self.output_size, device=device, dtype=torch.float),
                              torch.arange(self.output_size, device=device, dtype=torch.float),
                              indexing='ij')  # (output_size, output_size)
        center_out = (self.output_size - 1) / 2.0
        dx = i - center_out
        dy = j - center_out

        # 计算极坐标
        r_out = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dy, dx)

        # 计算输入半径r_in
        mask = r_out > self.center_size / 2
        delta_r = torch.where(mask, r_out - self.center_size / 2 , torch.zeros_like(r_out))
        # r_in = torch.where(mask, self.tau * (delta_r) ** 3 / 6 + r_out, r_out)
        r_in = torch.where(mask, self.center_size / 2 + (torch.exp(self.tau * (delta_r)) -1) / self.tau, r_out)

        # 转换为输入坐标
        center_in = (cropped_size - 1) / 2.0
        x_in = center_in + r_in * torch.cos(theta)
        y_in = center_in + r_in * torch.sin(theta)

        # 归一化到[-1, 1]
        x_norm = (x_in / (cropped_size - 1)) * 2 - 1
        y_norm = (y_in / (cropped_size - 1)) * 2 - 1

        # 生成采样网格
        grid = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).clamp(-1, 1)  # (B, output_size, output_size, 2)

        # 应用网格采样
        projected = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
        return projected


class EdgeDetectionLayer(nn.Module):
    def __init__(self, apply_gaussian: bool = False, gauss_kernel_size: int = 5, gauss_sigma: float = 1.0):
        super(EdgeDetectionLayer, self).__init__()
        # Sobel 算子：水平和竖直边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.apply_gaussian = apply_gaussian
        if apply_gaussian:
            self.gauss_kernel_size = gauss_kernel_size
            self.gauss_sigma = gauss_sigma
            # gaussian kernel will be created on-the-fly in forward to match device/dtype
        else:
            self.gauss_kernel_size = None
            self.gauss_sigma = None

    def forward(self, x):
        # 假设 x 的形状为 (B, C, H, W)
        # 对每个通道分别计算梯度
        B, C, H, W = x.shape

        if self.apply_gaussian:
            device = x.device
            dtype = x.dtype
            gauss = make_gaussian_kernel(self.gauss_kernel_size, float(self.gauss_sigma), device=device, dtype=dtype)
            # apply the same gaussian per-channel using groups=C
            gauss_per_channel = gauss.repeat(C, 1, 1, 1)  # (C,1,k,k)
            pad = self.gauss_kernel_size // 2
            x = F.conv2d(x, gauss_per_channel, padding=pad, groups=C)

        # 使用 conv2d 计算梯度
        grad_x = F.conv2d(x, self.sobel_x.to(x.device), padding=1, groups=1)
        grad_y = F.conv2d(x, self.sobel_y.to(x.device), padding=1, groups=1)
        grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad


class RGB2H(nn.Module):
    def __init__(self):
        super(RGB2H, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 3, "Input must be RGB image."
        r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)
        delta = max_val - min_val
        
        h_channel = torch.zeros_like(max_val)
        mask_rgb_equal = (delta < 1e-8)
        h_channel[mask_rgb_equal] = 0.0
        
        # 处理max为R的情况
        mask_max_r = (max_val == r) & (~mask_rgb_equal)
        h_channel[mask_max_r] = ((g[mask_max_r] - b[mask_max_r]) / delta[mask_max_r]) % 6
        
        # 处理max为G的情况
        mask_max_g = (max_val == g) & (~mask_rgb_equal)
        h_channel[mask_max_g] = ((b[mask_max_g] - r[mask_max_g]) / delta[mask_max_g]) + 2
        
        # 处理max为B的情况
        mask_max_b = (max_val == b) & (~mask_rgb_equal)
        h_channel[mask_max_b] = ((r[mask_max_b] - g[mask_max_b]) / delta[mask_max_b]) + 4
        
        h_channel = h_channel / 6.0
        h_channel = torch.clamp(h_channel, 0.0, 1.0)
        
        return h_channel.unsqueeze(1)


class FrameDifferenceLayer(nn.Module):
    '''
    差帧
    '''
    def __init__(self):
        super(FrameDifferenceLayer, self).__init__()
        self.x_old = None
        self.first_frame = True
    
    def forward(self, x):
        if self.first_frame:
            self.x_old = torch.zeros_like(x) 
            self.first_frame = False
        
        diff = torch.abs(x - self.x_old)
        self.x_old = x
        return diff


class OpticalFlowLayer(nn.Module):
    def __init__(self, lateralField=2, flowLayerCache=5, tau=0.5):
        """
        初始化 LateralInfulentialOpticalFlowLayer 模块。

            参数:
                LateralField(int): LateralField 的半径，光流节点的感受野，默认值为 2
                flowLayerCache(int): 缓存帧数，默认值为 5
                tau(float): 信号衰减
        """
        super(OpticalFlowLayer, self).__init__()
        self.lateralField = lateralField
        self.flowLayerCache = flowLayerCache
        self.tau = tau
        self.threshold = None
        
        assert flowLayerCache > 2 * lateralField, "flowLayerCache and lateralField has to satisfy: flowLayerCache >= 2 * lateralField + 1"
        
        self.cache = []
        self.cacheforMaxpooling = []
        self.offsets = [dx for dx in range(-lateralField, lateralField+1)] # frame = -lF -> 0 -> lF  ,2*lF+1 in total
        
        self.valid = None
        self.j = None
        self.i = None
    
    def gird_precompute(self, H, W):
        '''
        计算偏移网格
        '''

        valid = torch.tensor([False] * H * W * (2 * self.lateralField + 1) * 4, dtype=torch.bool).reshape(H, W, 2 * self.lateralField + 1, 4)
        j ,i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        for dx in range(-self.lateralField, self.lateralField+1):
            grid_in_x = i + dx # x
            grid_in_y = j + dx # y
            grid_out_x = i - dx # x
            grid_out_y = j - dx # y     
            
            valid[...,dx + self.lateralField, 0] = (grid_in_x >= 0) & (grid_in_x < W)
            valid[...,dx + self.lateralField, 1] = (grid_in_y >= 0) & (grid_in_y < H)
            valid[...,dx + self.lateralField, 2] = (grid_out_x >= 0) & (grid_out_x < W)
            valid[...,dx + self.lateralField, 3] = (grid_out_y >= 0) & (grid_out_y < H)
            
        return valid, j, i
    
    def forward(self, x, max_in_field):
        """
        计算光流的xy分量。

            Warning:
                仅支持单批次输入
            
            输入:
                支持:
                    grad_x (torch.Tensor): (B, 1, H, W) + grad_y (torch.Tensor): (B, 1, H, W)
                    grad (torch.Tensor): (B, 1, H, W)
                    RGB (torch.Tensor): (B, 3, H, W)
                            B == 1
            输出:
                sum_influence_x, sum_influence_y
        """
        
        # 输入 -> (B, 1, H, W)
        dim = len(x)
        if dim == 2:
            grad_x, grad_y = x
            B, C, H, W = grad_x.shape
            x = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        elif dim == 1:
            B, C, H, W = x.shape
            # if C == 3:
            #     togray = T.Grayscale()
            #     x = togray(x)
        assert B == 1, "Only One Batch Once"
        
        if self.valid is None:
            self.valid, self.j, self.i = self.gird_precompute(H, W)
        
        # influence by this frame
        influence = torch.zeros_like(x).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, self.lateralField*2+1, 2)
        # influence = torch.zeros(1, 1, H, W, self.lateralField*2+1, 2)
        for dx_idx, dx in enumerate(self.offsets):
            
            mask_x = self.valid[..., dx_idx, 0]
            mask_y = self.valid[..., dx_idx, 1]
            
            i_shifted = self.i + dx
            j_shifted = self.j + dx
            
            for c in range(C):
                strength_x = torch.zeros_like(x[0, 0])
                strength_y = torch.zeros_like(x[0, 0])
                
                strength_x[mask_x] = x[0, c, self.j[mask_x], i_shifted[mask_x]]
                strength_y[mask_y] = x[0, c, j_shifted[mask_y], self.i[mask_y]]
                
                influence[0, c, :, :, dx_idx, 0] = strength_x.unsqueeze(0).unsqueeze(0)  
                influence[0, c, :, :, dx_idx, 1] = strength_y.unsqueeze(0).unsqueeze(0)
        
        
        # stack of influence
        # (1, C, H, W, self.lateralField * 2 + 1, 2) * self.flowLayerCache  --stack-->  (1, C, H, W, self.lateralField * 2 + 1, 2, self.flowLayerCache) 
        #  B  C  H  W  - lateralField -> dx -> lateralField    horizontal/vertical      1 -> frame -> flowLayerCache
        # (1, C, H, W,        self.lateralField * 2 + 1,               2,                   self.flowLayerCache) 
        self.cache.append(influence)
        self.cacheforMaxpooling.append(max_in_field)
        
        sum_influence_x = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        sum_influence_y = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        # sum_influence_x = torch.zeros(1, 1, H, W, 2)
        # sum_influence_y = torch.zeros(1, 1, H, W, 2)
        max_in_fieldandtime = torch.zeros_like(x)
        flow = torch.zeros_like(x).unsqueeze(-1).repeat(1, 1, 1, 1, 4)

        frames = len(self.cache)
        assert frames == len(self.cacheforMaxpooling), "Max pooling layer has to be engaged in the network at the same time as the flow layer"
        if frames > self.flowLayerCache:
            self.cache.pop(0)
            self.cacheforMaxpooling.pop(0)
            imCache = torch.stack(self.cache, dim = -1)
            poolingCache = torch.stack(self.cacheforMaxpooling, dim = -1)
            
            for sample in range(0, self.lateralField * 2 + 1):
                sum_influence_x[0, :, :, :, 0] += imCache[0, :, :, :, sample, 0, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_x[0, :, :, :, 1] += imCache[0, :, :, :, sample, 0, self.flowLayerCache - sample - 1]
                sum_influence_y[0, :, :, :, 0] += imCache[0, :, :, :, sample, 1, self.flowLayerCache - self.lateralField * 2 + sample - 1]
                sum_influence_y[0, :, :, :, 1] += imCache[0, :, :, :, sample, 1, self.flowLayerCache - sample - 1]
            
            max_result = torch.max(poolingCache, dim=4)
            max_in_fieldandtime = max_result.values 
            flow[..., 0] = torch.relu(sum_influence_x[0, :, :, :, 0] - max_in_fieldandtime)
            flow[..., 1] = torch.relu(sum_influence_x[0, :, :, :, 1] - max_in_fieldandtime)
            flow[..., 2] = torch.relu(sum_influence_y[0, :, :, :, 0] - max_in_fieldandtime)
            flow[..., 3] = torch.relu(sum_influence_y[0, :, :, :, 1] - max_in_fieldandtime)
            
        else: print("not Enough Frames")
                
        factor = torch.exp(-self.tau * torch.ones_like(influence))
        self.cache = [tensor * factor for tensor in self.cache]

        return flow


def smooth_moving(input1, input2, velocity):
    center_x0, center_y0 = input1
    center_x1, center_y1 = input2
    dx = center_x1 - center_x0
    dy = center_y1 - center_y0
    distance = math.sqrt(dx**2 + dy**2)
    steps = int(distance // velocity)
    x = torch.linspace(center_x0, center_x1, steps+1)
    y = torch.linspace(center_y0, center_y1, steps+1)
    return x, y


class RetinaModel(nn.Module):
    def __init__(self, cropped_size=512, output_size=512, center_size=256, projectiontau=0.01,
                 lateralField=2, flowLayerCache=5, flowLayertau=0.5,
                 edge_apply_gaussian: bool = False, edge_gauss_kernel_size: int = 5, edge_gauss_sigma: float = 1.0):
        '''
        较好的projectiontau
            cropped_size = 1024
            output_size = 512
            center_size = 256
            tau = 0.01
        
        Added optional gaussian smoothing before the edge/sobel convs. If you want
        the advanced_feature pipeline to match an upstream gaussian blur, enable
        `edge_apply_gaussian` and set `edge_gauss_kernel_size`/`edge_gauss_sigma`.
        '''
        super().__init__()
        self.preprocess = PreprocessLayer(cropped_size)
        self.projection = ProjectionLayer(output_size, center_size, projectiontau)
        # pass gaussian options to edge detection to allow matching advanced_feature's smoothing
        self.edge_detection = EdgeDetectionLayer(apply_gaussian=edge_apply_gaussian,
                                                 gauss_kernel_size=edge_gauss_kernel_size,
                                                 gauss_sigma=edge_gauss_sigma)
        self.rgb2h = RGB2H()
        self.frame_diff = FrameDifferenceLayer()
        self.maxpooling_for_flowlayer = nn.MaxPool2d(kernel_size=lateralField * 2 + 1, stride=1, padding=lateralField)
        self.flow = OpticalFlowLayer(lateralField, flowLayerCache, flowLayertau)
        
    def forward(self, x, center_x, center_y):
        cropped = self.preprocess(x, center_x, center_y)          # [1, C, H0, W0]
        x = cropped
        # x = self.projection(cropped)                              # [1, C, H, W]
        grad = self.edge_detection(x)                       # [1, 1, H, W, 2]
        h = self.rgb2h(x)                                   # [1, 1, H, W]
        diff = torch.zeros_like(h)
        # diff = self.frame_diff(x)                           # [1, C, H, W]
        flowvelocity = torch.zeros_like(h)
        # max_in_field = self.maxpooling_for_flowlayer(diff)  #
        # flowvelocity = self.flow(diff, max_in_field)        # [1, C, H, W, 4]
        
        # Return `cropped` as well so advanced_feature can accept the preprocessed tensor
        return x, grad, h, diff, flowvelocity, cropped
