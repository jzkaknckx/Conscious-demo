import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

class SWTModule(nn.Module):
    """
    Stationary Wavelet Transform (SWT) Module.
    Performs multi-scale decomposition without downsampling.
    Output shape: [B, 3J+1, H, W]
    Organization: [A_J, D_1^H, D_1^V, D_1^D, ..., D_J^H, D_J^V, D_J^D]
    """
    def __init__(self, J: int = 4, wavelet: str = 'haar', device: Optional[torch.device] = None):
        super().__init__()
        self.J = J
        self.wavelet = wavelet
        self.device = device or torch.device('cpu')
        
        # Standard Haar filters
        if wavelet == 'haar':
            h = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
            g = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
        else:
            # Default to Haar if unknown
            h = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
            g = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
            
        self.register_buffer('h', h)
        self.register_buffer('g', g)
        
        # Pre-compute 2D kernels
        # LL, LH, HL, HH
        ll = torch.outer(h, h)
        lh = torch.outer(h, g)
        hl = torch.outer(g, h)
        hh = torch.outer(g, g)
        
        self.register_buffer('kernels', torch.stack([ll, lh, hl, hh]).unsqueeze(1)) # [4, 1, 2, 2]

    def _get_upsampled_filter(self, kernel: torch.Tensor, level: int) -> torch.Tensor:
        """Upsample filter by inserting 2^(level-1)-1 zeros between coefficients (a trous)."""
        if level == 1:
            return kernel
        
        step = 2**(level - 1)
        k_size = kernel.shape[-1]
        new_size = (k_size - 1) * step + 1
        
        upsampled = torch.zeros((kernel.shape[0], kernel.shape[1], new_size, new_size), device=self.device)
        for i in range(k_size):
            for j in range(k_size):
                upsampled[:, :, i * step, j * step] = kernel[:, :, i, j]
        return upsampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C, H, W]
        Output: [B, 3J+1, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. Grayscale conversion if needed
        if C >= 3:
            # Standard luminance weights: 0.299R + 0.587G + 0.114B
            weights = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(1, 3, 1, 1)
            img = torch.sum(x[:, :3, :, :] * weights, dim=1, keepdim=True)
        else:
            img = x[:, 0:1, :, :]
            
        current_a = img
        details = []
        
        for j in range(1, self.J + 1):
            # Get upsampled filters for level j
            kernels_j = self._get_upsampled_filter(self.kernels, j)
            pad = kernels_j.shape[-1] // 2
            
            # Apply convolution
            # We use circular padding for SWT to avoid boundary artifacts
            res = F.conv2d(F.pad(current_a, (pad, pad, pad, pad), mode='circular'), kernels_j)
            
            # If kernel size is even, F.pad circular might need adjustment for exact SWT
            # but for Haar [2,2] upsampled, pad=1 is usually fine. 
            # Adjusting to match spatial size exactly:
            res = res[:, :, :H, :W]
            
            a_j = res[:, 0:1, :, :]
            h_j = res[:, 1:2, :, :]
            v_j = res[:, 2:3, :, :]
            d_j = res[:, 3:4, :, :]
            
            details.append(h_j)
            details.append(v_j)
            details.append(d_j)
            
            current_a = a_j
            
        # Output: [A_J, D_1^H, D_1^V, D_1^D, ..., D_J^H, D_J^V, D_J^D]
        output = torch.cat([current_a] + details, dim=1)
        return output
