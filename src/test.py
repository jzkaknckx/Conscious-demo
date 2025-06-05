import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math
from typing import List, Optional, Tuple, Union, Any, Dict

from trail.pic_tools import *
from trail.matrix_tools import *

from nn.Retina_d_3rd import (
    PreprocessLayer
    ,ProjectionLayer
    ,EdgeDetectionLayer
)

x = torch.ones(1,1,20,20)
for i in range(20):
    for j in range(20):
        x[0,0,i,j] = i + j*0.1

maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
y = maxpool(x)

print(x)
print(y)