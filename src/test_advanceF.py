import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from collections import deque

from trail.pic_tools import *
from trail.matrix_tools import *

torch.set_printoptions(profile="full",linewidth=512)

from nns.Retina import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
    ,RetinaModel
)

from nns.advanced_feature_enhanced import * 


if __name__ == "__main__":
    fig, axes = plt.subplots(3,3, figsize=(10, 5))
    
    # 准备输入
    image_path = 'src/picture/trail/pattern_tria.png'  # [1, 3, 1280, 1920]
    tensor, oringinal_image = image_to_tensor(image_path)
    axes[0,0].imshow(tensor_to_image(tensor))
    '''
    # curve test below
    # curvFilter = CurvatureFilter()
    # curve = curvFilter(tensor)
    curvFilters = MultiScaleFeatureBank([1.0, 2.0, 4.0])
    curve = curvFilters(tensor)[1]
    
    axes[0,0].imshow(tensor_to_image(tensor))
    axes[1,0].imshow(tensor_to_image(curve[:, 0, :, :]))
    axes[1,1].imshow(tensor_to_image(curve[:, 1, :, :]))
    axes[1,2].imshow(tensor_to_image(curve[:, 2, :, :]))
    
    # scaleagg1 = ScaleAggregator(3, 1, 'attn') 
    # scaleagg2 = ScaleAggregator(3, 1, 'max')
    # scaleagg3 = ScaleAggregator(3, 1, 'conv')
    
    # curve1 = scaleagg1(curve)
    # curve2 = scaleagg2(curve)
    # curve3 = scaleagg3(curve)
    # axes[2,0].imshow(tensor_to_image(curve1))
    # axes[2,1].imshow(tensor_to_image(curve2))
    # axes[2,2].imshow(tensor_to_image(curve3))
    '''
    '''
    #rectangle test below
    rectFilterM = MultiScaleFeatureBank([2.0, 5.0, 10.0])
    rec = rectFilterM(tensor)[1]
    rec0 = rec[:, 0, :, :].unsqueeze(1)
    rec1 = rec[:, 1, :, :].unsqueeze(1)
    rec2 = rec[:, 2, :, :].unsqueeze(1)
    axes[1,0].imshow(edge_to_image(rec0))
    axes[1,1].imshow(edge_to_image(rec1))
    axes[1,2].imshow(edge_to_image(rec2))
    
    recagg1 = AspectAggregator(3, 1, 'attn') 
    recagg2 = AspectAggregator(3, 1, 'max')
    recagg3 = AspectAggregator(3, 1, 'conv')
    
    reca1 = recagg1(rec)
    reca2 = recagg2(rec)
    reca3 = recagg3(rec)

    axes[2,0].imshow(tensor_to_image(reca1))
    axes[2,1].imshow(edge_to_image(reca2))
    axes[2,2].imshow(tensor_to_image(reca3))
    '''
    
    #orientation test below
    oriFilterM = MultiScaleFeatureBank([1.0, 5.0, 10.0])
    rec = oriFilterM(tensor)[2]
    rec0 = rec[:, 0, :, :].unsqueeze(1)
    rec1 = rec[:, 1, :, :].unsqueeze(1)
    rec2 = rec[:, 2, :, :].unsqueeze(1)
    axes[1,0].imshow(edge_to_image(rec0))
    axes[1,1].imshow(edge_to_image(rec1))
    axes[1,2].imshow(edge_to_image(rec2))
    recagg1 = AspectAggregator(3, 1, 'attn') 
    recagg2 = AspectAggregator(3, 1, 'max')
    recagg3 = AspectAggregator(3, 1, 'conv')
    
    reca1 = recagg1(rec)
    reca2 = recagg2(rec)
    reca3 = recagg3(rec)

    axes[2,0].imshow(tensor_to_image(reca1))
    axes[2,1].imshow(edge_to_image(reca2))
    axes[2,2].imshow(tensor_to_image(reca3))
    
    plt.show()