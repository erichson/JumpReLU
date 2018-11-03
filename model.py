import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from activationfun import *


class NetW(nn.Module):
    def __init__(self):
        super(NetW, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),# 32x32x3 -> 32x32x64
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
            nn.Dropout()
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50,10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class JumpNetW(nn.Module):
    def __init__(self):
        super(JumpNetW, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),# 32x32x3 -> 32x32x64
            JumpReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
            JumpReLU(),
            nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
            nn.Dropout()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            JumpReLU(),
            nn.Linear(50,10),
        )



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x






class NetW_EMNIST(nn.Module):
    def __init__(self):
        super(NetW_EMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),# 32x32x3 -> 32x32x64
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
            nn.Dropout()
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 100),
            nn.ReLU(),
            nn.Linear(100, 48),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class JumpNetW_EMNIST(nn.Module):
    def __init__(self):
        super(JumpNetW_EMNIST, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1,3, kernel_size=1), # 1 to 1 map
            JumpReLU(),
            nn.Conv2d(3, 10, kernel_size=5),# 32x32x3 -> 32x32x64
            JumpReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
            JumpReLU(),
            nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
            nn.Dropout()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(320, 100),
            JumpReLU(),
            nn.Linear(100, 48),
        )



    def forward(self, x):
        print(x.shape)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x