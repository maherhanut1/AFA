import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import argparse
import os
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

from models.layers.rAFA_conv import Conv2d as AFAConv




def replace_conv_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Extract parameters from the existing Conv2d layer
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias is not None
            padding_mode = module.padding_mode
            
            # Create a new custom conv layer with the same parameters
            new_conv = AFAConv(
                in_channels,
                out_channels,
                kernel_size[0],
                rank=out_channels//2,
                stride=stride,
                padding=padding,
                dilation=dilation,
                padding_mode=padding_mode,
                groups=groups
            )
            
            # Optionally, copy the weights and bias from the original layer
            with torch.no_grad():
                new_conv.weight = nn.Parameter(module.weight.clone())
                if bias:
                    new_conv.bias = nn.Parameter(module.bias.clone())
            
            # Replace the original Conv2d layer with the new custom layer
            setattr(model, name, new_conv)
        else:
            # Recursively apply to child modules
            replace_conv_layers(module)



model = models.resnet50()
replace_conv_layers(model)
out = model(torch.randn((1, 3, 244, 244)))
loss = out.mean()
loss.backward()