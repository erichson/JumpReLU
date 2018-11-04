import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from activationfun import *


# class JumpNet(nn.Module):
#     def __init__(self, jump_val = 0.):
#         super(JumpNet, self).__init__()
#         self.jump = jump_val
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 3, kernel_size=1),# 32x32x3 -> 32x32x64
#             JumpReLU(self.jump),                
#             nn.Conv2d(3, 10, kernel_size=5),# 32x32x3 -> 32x32x64
#             JumpReLU(self.jump),
#             nn.MaxPool2d(2),
#             nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
#             JumpReLU(self.jump),
#             nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
#             nn.Dropout(self.jump)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(320, 50),
#             JumpReLU(self.jump),
#             nn.Linear(50,10),
#         )



#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
        
#         return x



class JumpNet(nn.Module):
    def __init__(self, jump_val=0.):
        super(NetW_shift, self).__init__()
        
        self.jump = jump_val
        self.conv1 = nn.Conv2d(1,3, kernel_size=1) # 1 to 1 map
        self.relu1 = JumpReLU(self.jump)
        self.conv2 = nn.Conv2d(3, 10, kernel_size=5)# 32x32x3 -> 32x32x64
        self.relu2 = JumpReLU(self.jump)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5)# 16x16x64 -> 16x16x64
        self.relu3 = JumpReLU(self.jump)
        self.pool2 = nn.MaxPool2d(2)# 16x16x64 -> 8x8x64
        self.drop1 = nn.Dropout()

        self.linear1 = nn.Linear(320, 50)
        self.relu4 = JumpReLU(self.jump)
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
    def __init__(self):
        super(JumpNet_EMNIST, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),# 32x32x3 -> 32x32x64
            JumpReLU(self.jump), 
            nn.Conv2d(3, 10, kernel_size=5),# 32x32x3 -> 32x32x64
            JumpReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=5),# 16x16x64 -> 16x16x64
            JumpReLU(self.jump),
            nn.MaxPool2d(2),# 16x16x64 -> 8x8x64
            nn.Dropout()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(320, 200),
            JumpReLU(self.jump),
            nn.Linear(200, 48),
        )



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
