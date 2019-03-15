import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from activationfun import *

# LeNetLike Model definition
class LeNetLike(nn.Module):
    def __init__(self, jump=0.0):
        super(LeNetLike, self).__init__()
        
        self.jump = jump
        self.relu_jump = JumpReLU(jump=self.jump)
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 90)
        self.fc3 = nn.Linear(90, 10)


    def forward(self, x):
        x = self.relu_jump(F.max_pool2d(self.conv1(x), 2))
        x = self.relu_jump(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.dropout(x, training=self.training)        
        x = self.relu_jump(self.fc1(x))
        #x = F.dropout(x, training=self.training)                
        x = self.relu_jump(self.fc2(x))
        x = self.fc3(x)        
        return x
