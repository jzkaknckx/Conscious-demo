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

decayWithTime = [math.exp(- (t) * 1) for t in range(1, 10)]
x = [t for t in range(1, 10)]

print(x)