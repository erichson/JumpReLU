import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from activationfun import *

# AlexLike Model definition
class AlexLike(nn.Module):
    def __init__(self, jump=0.0):
        super(AlexLike, self).__init__()
        
        self.jump = jump
        self.relu_jump = JumpReLU(jump=self.jump)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        
        self.fc1 = nn.Linear(5*5*128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)


    def forward(self, x):
        x = self.relu_jump(self.conv1(x))
        x = F.max_pool2d(self.relu_jump(self.conv2(x)), 2)
        
        x = self.relu_jump2(self.conv3(x))
        x = F.max_pool2d(self.relu_jump(self.conv4(x)), 2)

        x = x.view(x.size(0), -1)

        x = F.dropout(x, training=self.training)        
        x = self.relu_jump(self.fc1(x))
        #x = F.dropout(x, training=self.training)                
        x = self.relu_jump(self.fc2(x))
        x = self.fc3(x)
        return x   
    
    
