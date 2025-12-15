import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import math
import heapq

# ---------------------------
# High-order feature detector
# ---------------------------
class HighOrderFeatureLayer(nn.Module):
    """
    输入:
        grad: 边缘信息，支持以下两种格式：
            - [B, 2, H, W]  (grad_x, grad_y)
            - [B, 1, H, W, 2] 或 [B, H, W, 2]
    输出:
        tensor of shape [B, 2, H, W]  -> channel0: curvature, channel1: rectangular response
    说明:
        - 低复杂度实现：Laplacian-like 2阶差分估计曲率；Sobel-based方向强度用于长条矩形响应。
    """
    def __init__(self):
        super().__init__()
        # 固定卷积核（以注册buf的形式保留在设备上）
        lap = torch.tensor([[0., -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]], dtype=torch.float32).view(1,1,3,3)
        sobel_x = torch.tensor([[-1.,0,1],
                                 [-2,0,2],
                                 [-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = sobel_x.transpose(2,3)
        self.register_buffer('laplacian', lap)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _normalize_grad_input(self, grad):
        # normalize various possible grad shapes -> [B,2,H,W]
        if grad is None:
            raise ValueError("grad must be provided")
        if grad.dim() == 5 and grad.size(-1) == 2:
            # [B,1,H,W,2]  => to [B,2,H,W]
            grad = grad.squeeze(1).permute(0,3,1,2).contiguous()
        elif grad.dim() == 4 and grad.size(1) == 2:
            grad = grad
        elif grad.dim() == 4 and grad.size(-1) == 2:
            # [B, H, W, 2]
            grad = grad.permute(0,3,1,2).contiguous()
        else:
            raise ValueError(f"Unsupported grad shape {grad.shape}")
        return grad

    def forward(self, grad):
        # grad -> [B,2,H,W]
        g = self._normalize_grad_input(grad)
        gx = g[:,0:1,...]
        gy = g[:,1:2,...]
        # gradient magnitude
        mag = torch.sqrt(gx*gx + gy*gy + 1e-6)  # [B,1,H,W]

        # curvature response: laplacian on mag (simple proxy for curvature)
        curvature = F.conv2d(mag, self.laplacian, padding=1)
        curvature = F.relu(curvature)  # 非负响应，曲率越大值越大

        # rectangular (long aspect) response:
        # idea: long thin rectangle -> one directional gradient dominates over neighborhood.
        # use sobel and local smoothing: max(|sobel_x|,|sobel_y|) followed by a local length-aware weighting
        rx = F.conv2d(mag, self.sobel_x, padding=1)
        ry = F.conv2d(mag, self.sobel_y, padding=1)
        rect = torch.max(rx.abs(), ry.abs())  # 强方向性响应
        # amplify responses where directional extent is long: approximate by local max pooling minus local min pooling
        # (cheap proxy for elongated shape)
        local_max = F.max_pool2d(rect, kernel_size=7, stride=1, padding=3)
        local_min = -F.max_pool2d(-rect, kernel_size=7, stride=1, padding=3)
        elongation = (local_max - local_min)  # 如果邻域内响应跨度大，表示有长结构
        rectangle = F.relu(rect * (1.0 + 0.5 * (elongation / (local_max + 1e-6))))

        out = torch.cat([curvature, rectangle], dim=1)  # [B,2,H,W]
        return out


# ---------------------------
# Feature fusion helper
# ---------------------------
class FeatureFusion(nn.Module):
    """
    将多通道信息 (edges, hue, diff, flow, highorder) 融合为：
      - 全局描述符 vector (用于快速全局匹配)
      - 下采样的局部描述 map (用于局部扩散匹配)
    设计要点（保持低复杂度）：
      - 每个输入通道先 L2 归一化（或 per-channel stdnorm）
      - 对每个通道做相同大小的下采样 (adaptive_avg_pool2d -> grid_h x grid_w)
      - 将各通道的下采样结果 concat -> flatten -> 线性降维投影（可选）
    输出:
      - global_vec: [D]  (float tensor, L2 normalized)
      - local_map: [C_local, grid_h, grid_w] (float tensor，用于局部匹配)
    """
    def __init__(self, grid_size=(8,8), proj_dim=256, use_projection=True):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.proj_dim = proj_dim
        self.use_projection = use_projection
        # 线性投影：把 concat 后的向量映射到较小维度
        if use_projection:
            # projection implemented as a 1x1 conv on the concatenated local map to preserve spatial shape
            # we'll initialize lazily after we see channel count
            self.proj_conv = None
        else:
            self.proj_conv = None

    def _prepare_tensor(self, t):
        # t: tensor [B,C,H,W] or [B,1,H,W]
        if t is None:
            return None
        # reduce channel dim to something manageable: if multiple channels, collapse with mean
        if t.dim() == 4:
            return t
        else:
            raise ValueError("unsupported tensor dims in fusion: {}".format(t.shape))

    def forward(self, edge_grad, hue, diff, flow, highorder):
        # edge_grad: could be [B,2,H,W] or [B,1,H,W,2] etc. We'll expect user passes mag or two-channel grad
        # Convert inputs to [B, C_i, H, W]
        # For edge, produce mag
        e = None
        if edge_grad is not None:
            if edge_grad.dim() == 5 and edge_grad.size(-1) == 2:
                # [B,1,H,W,2]
                eg = edge_grad.squeeze(1).permute(0,3,1,2).contiguous()
                e = torch.sqrt(eg[:,0:1]**2 + eg[:,1:2]**2 + 1e-6)  # [B,1,H,W]
            elif edge_grad.dim() == 4 and edge_grad.size(1) == 2:
                eg = edge_grad
                e = torch.sqrt(eg[:,0:1]**2 + eg[:,1:2]**2 + 1e-6)
            elif edge_grad.dim() == 4 and edge_grad.size(1) == 1:
                e = edge_grad
            else:
                raise ValueError("edge_grad shape not supported: {}".format(edge_grad.shape))
        # hue: [B,1,H,W], diff: [B,Cd,H,W], flow: [B,Cf,H,W] or [B,Cf,H,W,4] -> convert to magnitude
        h = hue
        d = diff
        f = flow
        if f is not None and f.dim() == 5 and f.size(-1) == 4:
            # last dim might be flow vectors; collapse
            # [B, C, H, W, 4] -> compute magnitude across last dim -> [B,C,H,W]
            f = torch.sqrt((f**2).sum(dim=-1) + 1e-6).permute(0,1,2,3) if f.dim()==5 else f
            # But the above permute is redundant; simpler:
            f = torch.sqrt((f**2).sum(dim=-1) + 1e-6)
        # Ensure d,f are [B, C, H, W]
        # For simplicity, collapse multi-channel diff/flow by channel-mean to keep descriptor size small
        if d is not None:
            if d.dim() == 4 and d.size(1) > 1:
                d = d.mean(dim=1, keepdim=True)
        if f is not None:
            if f.dim() == 4 and f.size(1) > 1:
                f = f.mean(dim=1, keepdim=True)

        # collect available channels
        channels = []
        if e is not None: channels.append(e)
        if h is not None: channels.append(h)
        if d is not None: channels.append(d)
        if f is not None: channels.append(f)
        if highorder is not None: channels.append(highorder)  # highorder is [B,2,H,W]

        if len(channels) == 0:
            raise ValueError("No channels provided to fusion")

        # concat along channel axis
        x = torch.cat(channels, dim=1)  # [B, C_all, H, W]
        B, C_all, H, W = x.shape

        # per-channel normalize (L2 over spatial)
        # to avoid dividing by tiny values, add eps
        eps = 1e-6
        norms = torch.sqrt((x.view(B, C_all, -1)**2).sum(-1, keepdim=True) + eps)  # [B,C,1]
        x_norm = x / (norms.view(B, C_all, 1, 1) + eps)

        # create downsampled local map
        local_map = F.adaptive_avg_pool2d(x_norm, (self.grid_h, self.grid_w))  # [B,C_all,gh,gw]

        # global descriptor: flatten then L2 normalize
        global_vec = local_map.view(B, -1)  # [B, C_all*gh*gw]
        # optional projection (1x1 conv on local_map)
        if self.use_projection:
            if self.proj_conv is None:
                # lazy init
                self.proj_conv = nn.Conv2d(C_all, self.proj_dim, kernel_size=1)
                # move to same device as inputs
                self.proj_conv = self.proj_conv.to(x.device)
            proj_map = self.proj_conv(local_map)  # [B, proj_dim, gh, gw]
            global_vec = proj_map.view(B, -1)
        # L2 normalize global vector
        gnorm = torch.norm(global_vec, p=2, dim=1, keepdim=True) + 1e-6
        global_vec = global_vec / gnorm

        return global_vec, local_map  # local_map still normalized per-channel


# ---------------------------
# Memory pool
# ---------------------------
class MemoryPool:
    """
    经验池（非 nn.Module）：
      - 储存记忆神经元（原型）
      - 每个原型包含:
            * prototype_vec (全局描述向量)
            * local_map (下采样的局部描述，用于局部匹配) - torch tensor
            * strength (activation count)
            * age (帧数未匹配计数)
            * connections (可选，指向上一层神经元的索引/位置信息)
            * meta (可存 scale 等信息)
      - 接口:
            insert_or_update(global_vec, local_map, meta)
            query(global_vec, local_map) -> returns matched index or None
            step_aging() -> 增加 age，并清理老化神经元
    参数说明:
      - global_thresh: 全局匹配阈值 (越小越严格)
      - local_thresh: 局部像素差阈值
      - local_radius: 局部扩散搜索半径（在下采样网格单元上）
      - min_local_matches: 扩散找到连通区域的最小像素数
      - top_k_check: 为减少复杂度，仅检查 top_k 个 prototype（按 strength 排序）
    """
    def __init__(self,
                 device='cpu',
                 global_thresh=0.3,
                 local_thresh=0.2,
                 local_radius=1,
                 min_local_matches=6,
                 age_limit=10,
                 top_k_check=50):
        self.protos = []  # list of dicts
        self.device = device
        self.global_thresh = global_thresh
        self.local_thresh = local_thresh
        self.local_radius = local_radius
        self.min_local_matches = min_local_matches
        self.age_limit = age_limit
        self.top_k_check = top_k_check

    def _to_cpu(self, t):
        return t.detach().cpu() if isinstance(t, torch.Tensor) else t

    def _global_distance(self, a, b):
        # a, b are 1D torch vectors L2-normalized
        # use Euclidean distance (equivalent to sqrt(2-2*cos))
        return torch.norm(a - b).item()

    def _local_match_by_diffusion(self, cur_local, mem_local):
        """
        cur_local, mem_local: [C, gh, gw] (torch tensors on cpu or same device)
        returns True if a local connected region with size >= min_local_matches found
        algorithm:
          - compute per-pixel L2 difference across channels (result [gh,gw])
          - create binary mask where diff < local_thresh
          - find connected components (4-neigh) and check max component size >= min_local_matches
        """
        # move to cpu numpy for easy BFS operations (small grids)
        if isinstance(cur_local, torch.Tensor):
            cur_local = cur_local.detach().cpu().numpy()
        if isinstance(mem_local, torch.Tensor):
            mem_local = mem_local.detach().cpu().numpy()
        # compute per-pixel L2 diff across channels
        diff_map = ((cur_local - mem_local)**2).sum(axis=0)  # [gh, gw]
        diff_map = (diff_map**0.5)
        mask = diff_map < self.local_thresh  # boolean mask
        gh, gw = mask.shape
        visited = [[False]*gw for _ in range(gh)]
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        for i in range(gh):
            for j in range(gw):
                if mask[i,j] and not visited[i][j]:
                    # BFS
                    q = deque()
                    q.append((i,j))
                    visited[i][j] = True
                    cnt = 0
                    while q:
                        ci,cj = q.popleft()
                        cnt += 1
                        # early stop if reach min matches
                        if cnt >= self.min_local_matches:
                            return True
                        for di,dj in dirs:
                            ni, nj = ci+di, cj+dj
                            if 0<=ni<gh and 0<=nj<gw and mask[ni,nj] and not visited[ni][nj]:
                                visited[ni][nj] = True
                                q.append((ni,nj))
        return False

    def query_and_update(self, global_vec, local_map, meta=None):
        """
        插入或更新：
          - 先按 strength 排序取 top_k_check 原型并计算全局距离，若 <= global_thresh 则强化并返回 idx
          - 否则对 top_k_check 原型做局部扩散匹配(local_map)，若命中则新建局部原型或强化现有原型
          - 若无任何匹配，则新建完整原型
        返回:
          - idx, action  (action in {'matched_global','matched_local','new'})
        """
        # ensure CPU tensors for small memory pool ops (reduce GPU-CPU overhead for many small ops)
        # but keep original proto tensors saved as cpu to reduce GPU mem footprint
        g = global_vec.detach().cpu()
        # prepare candidate prototypes (top-k by strength)
        if len(self.protos) == 0:
            # create new proto
            proto = {
                'global_vec': g.clone(),
                'local_map': local_map.detach().cpu().squeeze(0).clone(),  # [C,gh,gw]
                'strength': 1,
                'age': 0,
                'meta': meta
            }
            self.protos.append(proto)
            return len(self.protos)-1, 'new'

        # get indices of top-k by strength
        strengths = [p['strength'] for p in self.protos]
        top_k = heapq.nlargest(min(self.top_k_check, len(self.protos)), range(len(self.protos)), key=lambda i: strengths[i])

        # 1) global check
        best_idx = None
        best_dist = float('inf')
        for i in top_k:
            pg = self.protos[i]['global_vec']
            d = self._global_distance(g.view(-1), pg.view(-1))
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist <= self.global_thresh:
            # strengthen best prototype
            p = self.protos[best_idx]
            # running average update on global_vec (small memory footprint)
            p['global_vec'] = (p['global_vec'] * p['strength'] + g) / (p['strength'] + 1)
            p['strength'] += 1
            p['age'] = 0
            return best_idx, 'matched_global'

        # 2) local diffusion check
        # check local_map match for top_k prototypes
        cur_local = local_map.detach().cpu().squeeze(0)  # [C,gh,gw]
        for i in top_k:
            mem_local = self.protos[i]['local_map']
            if self._local_match_by_diffusion(cur_local.numpy(), mem_local.numpy()):
                # local match: create new prototype that stores only this local area?  
                # For simplicity here we strengthen mem proto and also create a short-local proto to specialize if desired.
                p = self.protos[i]
                # Strengthen
                p['strength'] += 1
                p['age'] = 0
                # Optionally create a local-specialized proto (store cur_local)
                # We'll create one with same global but local_map = cur_local to represent this local specialization
                local_proto = {
                    'global_vec': g.clone(),
                    'local_map': cur_local.clone(),
                    'strength': 1,
                    'age': 0,
                    'meta': meta
                }
                self.protos.append(local_proto)
                return len(self.protos)-1, 'matched_local_created'
        # 3) no match -> create new
        proto = {
            'global_vec': g.clone(),
            'local_map': cur_local.clone(),
            'strength': 1,
            'age': 0,
            'meta': meta
        }
        self.protos.append(proto)
        return len(self.protos)-1, 'new'

    def step_aging(self):
        # increment age for all prototypes not touched, and remove those older than limit
        new_list = []
        for p in self.protos:
            p['age'] += 1
            if p['age'] <= self.age_limit:
                new_list.append(p)
        self.protos = new_list

    def stats(self):
        return {'n_protos': len(self.protos), 'top_strengths': sorted([p['strength'] for p in self.protos], reverse=True)[:10]}

# ---------------------------
# Usage example (不改变原 RetinaModel)
# ---------------------------

# 假定你已有 retina = RetinaModel(...) 并得到输出:
# x, grad, h, diff, flowvelocity = retina(input_img, center_x, center_y)

# 新增层实例化（在模型初始化或外部脚本中）
# high_layer = HighOrderFeatureLayer().to(device)
# fusion = FeatureFusion(grid_size=(8,8), proj_dim=256, use_projection=True).to(device)
# mem = MemoryPool(device='cpu', global_thresh=0.3, local_thresh=0.2, min_local_matches=6, age_limit=12, top_k_check=50)

# 在推理循环中：
# high = high_layer(grad)  # [B,2,H,W]
# global_vec, local_map = fusion(edge_grad=grad, hue=h, diff=diff, flow=flowvelocity, highorder=high)
# idx, action = mem.query_and_update(global_vec, local_map, meta={'center':(center_x,center_y)})
# mem.step_aging()
