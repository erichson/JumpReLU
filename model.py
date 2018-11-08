import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from activationfun import *


# class JumpNet(nn.Module):
#     def __init__(self):
#         super(JumpNet, self).__init__()
        
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 3, kernel_size=1),# 32x32x3 -> 32x32x64
#             JumpReLU(),                
#             nn.Conv2d(3, 10, kernel_size=5),# 32x32x3 -> 32x32x64
#             JumpReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
#             JumpReLU(),
#             nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
#             nn.Dropout()
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(320, 50),
#             JumpReLU(),
#             nn.Linear(50,10),
#         )



#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
        
#         return x
    
class JumpNet(nn.Module):
    def __init__(self, shift=0.):
        super(JumpNet, self).__init__()
        
        self.shift = shift
        self.conv1 = nn.Conv2d(1,3, kernel_size=1) # 1 to 1 map
        self.relu1 = JumpReLU(shift=self.shift)
        self.conv2 = nn.Conv2d(3, 10, kernel_size=5)# 32x32x3 -> 32x32x64
        self.relu2 = JumpReLU(shift=self.shift)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5)# 16x16x64 -> 16x16x64
        self.relu3 = JumpReLU(shift=self.shift)
        self.pool2 = nn.MaxPool2d(2)# 16x16x64 -> 8x8x64
        self.drop1 = nn.Dropout()

        self.linear1 = nn.Linear(320, 50)
        self.relu4 = JumpReLU(shift=self.shift)
        self.linear2 = nn.Linear(50,10)
        
    def forward(self, x, measure=False, shift=0.):
        output = []        
        x = self.conv1(x)
        output.append(x.data+0.)
        x = self.conv2(self.relu1(x))
        output.append(x.data+0.)
        x = self.conv3(self.pool1(self.relu2(x)))
        output.append(x.data+0.)
        x = self.drop1(self.pool2(self.relu3(x)))
        x = x.view(x.size(0), -1)
        x = self.linear2(self.relu4(self.linear1(x)))
        
        if measure:
            return output
        
        return x



class JumpNet_EMNIST(nn.Module):
    def __init__(self, shift=0.):
        super(JumpNet_EMNIST, self).__init__()
        
        self.shift = shift
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),# 32x32x3 -> 32x32x64
            JumpReLU(shift=self.shift), 
            nn.Conv2d(3, 10, kernel_size=5),# 32x32x3 -> 32x32x64
            JumpReLU(shift=self.shift),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
            JumpReLU(shift=self.shift),
            nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
            nn.Dropout()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(320, 200),
            JumpReLU(shift=self.shift),
            nn.Linear(200, 48),
        )



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

    
    
class Net_CIFAR(nn.Module):
    def __init__(self, shift=0.):
        super(Net_CIFAR, self).__init__()
        
        self.shift = shift
        
        self.features = nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=1),# 32x32x3 -> 32x32x64
            nn.Conv2d(3, 64, kernel_size=3),# 32x32x3 -> 32x32x64
            JumpReLU(shift=self.shift), 
            nn.Conv2d(64, 64, kernel_size=3),# 32x32x3 -> 32x32x64
            JumpReLU(shift=self.shift), 
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),# 32x32x3 -> 32x32x64
            JumpReLU(shift=self.shift),
            nn.Conv2d(128, 128, kernel_size=3),
            JumpReLU(shift=self.shift),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(5*5*128, 256),
            JumpReLU(shift=self.shift),
            nn.Linear(256, 256),
            JumpReLU(shift=self.shift),
            nn.Linear(256, 10),
        )



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x