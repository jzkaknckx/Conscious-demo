import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
import torchvision.transforms as transforms
import math

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_printoptions(profile="full",linewidth=512)

tensor = torch.tensor([[1,1],
                       [0,0]], 
                      dtype=torch.float32
                      )
    # [2,2]



tensor = tensor.repeat(3,1,1)
transform = transforms.ToPILImage()

mins,index = torch.min(tensor,1)
maxs = torch.max(tensor)
print(mins,index,maxs)

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes[0,0].imshow(transform(tensor))
axes[0,0].set_title('Original')
plt.show()