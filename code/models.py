import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
##########################  common layers ###############################################################

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): 
        return x.view(x.shape[0], -1)   
    
    
###########################  models ###############################################################

class FCNNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FCNNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, (7,1), 1, (3,0), bias=False),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (5,1), 1, (2,0), bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3,1), 1, (1,0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            GlobalAvgPool2d())
        
        self.fc = nn.Sequential(
            FlattenLayer(),
            nn.Linear(128, n_classes))  
        
    def forward(self, x: torch.Tensor):
        feature = self.conv(x)
        output = self.fc(feature)
        return output
    
