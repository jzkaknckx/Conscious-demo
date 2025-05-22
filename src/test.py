import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math

from trail.pic_tools import *
from trail.matrix_tools import *

from nn.Retina_d_3rd import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
)


a = b = torch.ones(1, 1, 5, 5, 2)
c = a[0, 0, :, :, 0] - b[0, 0, :, :, 0]
d = a[:, :, :, :, 0] - b[:, :, :, :, 0]
e = a[..., 0] - b[..., 0]

print(c.shape)
print(d.shape)
print(e.shape)