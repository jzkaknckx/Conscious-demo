class LateralInfulentialOpticalFlowLayer(nn.Module):
    def __init__(self, lateralField=5, flowLayerCache=5, alpha=1.0, tau=1.0):
        """
        初始化 LateralInfulentialOpticalFlowLayer 模块。

            参数:
                LateralField (int): LateralField 的半径，光流节点的感受野，默认值为 5
                flowLayerCache(int): 缓存帧数，默认值为 5
                alpha (float): 暂时没用
                beta (float): 时间衰减参数, 默认值为 1.0
        """
        super(LateralInfulentialOpticalFlowLayer, self).__init__()
        self.lateralField = lateralField  
        self.flowLayerCache = flowLayerCache 
        self.alpha = alpha  
        self.tau = tau  

        self.absx = None
        self.xold = None
        self.firstFrame = True
        self.Cache = []
        
        self.decayWithTime = [math.exp(- (f) * tau) for f in range(0, flowLayerCache + 2 * lateralField + 1)] # 衰减序列参数预计算
        
        self.offsets = [dx for dx in range(-lateralField, lateralField+1)] # frame = -lF -> 0 -> lF (2*lF + 1 in total)

    def forward(self, x):
        """
        前向传播函数，计算光流的xy分量。

            Warning:
                仅支持单批次输入
            
            输入:
                支持:
                    grad_x (torch.Tensor): (B, 1, H, W) + grad_y (torch.Tensor): (B, 1, H, W)
                    grad (torch.Tensor): (B, 1, H, W)
                    RGB (torch.Tensor): (B, 3, H, W)
                            B == 1
            输出:
            
        """
        # 输入 -> (B, 1, H, W)
        togray = T.Grayscale()
        
        dim = len(x)
        if dim == 2:
            grad_x, grad_y = x
            B, C, H, W = grad_x.shape
            x = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        elif dim == 1:
            B, C, H, W = x.shape
            if C == 3:
                x = togray(x)
        
        assert B == 1, "Only One Batch Once"
        
        # initialize sum_influence_x and sum_influence_y
        sum_influence_x = sum_influence_y = torch.zeros(1, 1, H, W)  # (H, W)
        
        
        if self.firstFrame == True:   # 第一帧不运算
            self.firstFrame = False
            print("First Frame")
        else: 
            self.absx = torch.abs(x - self.xold)
            self.Cache.append(self.absx)
            runFrame = len(self.Cache)
            
            if runFrame > self.flowLayerCache:  
                self.Cache.pop(0)
                imCache = torch.stack(self.Cache, dim = -1)
                
                # sum for each frame -> sum for each dx
                for dx in self.offsets:

                    # 邻域像素的坐标 (k, l)
                    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                    k = i + dx
                    l = j + dx

                    valid_x = (k >= 0) & (k < H)
                    valid_y = (l >= 0) & (l < W)

                    # 对应x+dx和y+dx的强度
                    strength_kj = strength_il = torch.zeros(1, 1, H, W, self.flowLayerCache)
                    strength_kj[valid_x.view(1, 1, H, W)] = imCache[0, 0, k[valid_x], j[valid_x]]
                    strength_il[valid_y.view(1, 1, H, W)] = imCache[0, 0, i[valid_y], l[valid_y]]

                    influence_x = influence_y = torch.zeros(1, 1, H, W)
                    for f in range(self.flowLayerCache):
                        influence_x += self.decayWithTime[self.flowLayerCache - f + self.lateralField + dx] * strength_kj[:, :, :, :, f]
                        influence_y += self.decayWithTime[self.flowLayerCache - f + self.lateralField + dx] * strength_il[:, :, :, :, f]

                    sum_influence_x += influence_x
                    sum_influence_y += influence_y
            else:   print("not Enough Frames")
        
        self.xold = x
        
        return sum_influence_x, sum_influence_y