"""
Train robust models to demonstrate JumpReLU.
"""

from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
import os

from advfuns import *
from models import *

from progressbar import *

#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='MNIST Example')
#
parser.add_argument('--name', type=str, default='mnist', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='Input batch size for training (default: 64)')
#
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='Input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='Number of epochs to train (default: 90)')
#
parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='Learning rate (default: 0.01)')
#
parser.add_argument('--lr-decay', type=float, default=0.2, help='Learning rate ratio')
#
parser.add_argument('--lr-schedule', type=str, default='normal', help='Learning rate schedule')
#
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[30,60,80], help='Decrease learning rate at these epochs.')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
#
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='Weight decay (default: 5e-4)')
#
parser.add_argument('--arch', type=str, default='LeNetLike',  help='Choose the architecture')
#
parser.add_argument('--depth', type=int, default=20, help='Choose the depth of resnet')
#
parser.add_argument('--jump', type=float, default=0.0, metavar='E', help='Jump value')

#
args = parser.parse_args()



#==============================================================================
# set random seed to reproduce the work
#==============================================================================
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.name + '_result'):
    os.mkdir(args.name + '_result')

for arg in vars(args):
    print(arg, getattr(args, arg))

#==============================================================================
# get dataset
#==============================================================================
train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)
print('data is loaded')


#==============================================================================
# get model and optimizer
#==============================================================================
model_list = {
        'LeNetLike': LeNetLike(jump=args.jump),
        'AlexLike': AlexLike(jump=args.jump),
        'JumpResNet': JumpResNet(depth=args.depth, jump=args.jump),
        'MobileNetV2': MobileNetV2(jump=args.jump),      
}


model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')
print(model)

#==============================================================================
# Run
#==============================================================================
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size()[0] < args.batch_size:
            continue
        
        model.train()
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()*target.size()[0]
        total_num += target.size()[0]
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        
        
        optimizer.step()
        optimizer.zero_grad()
    
    # print progress        
    progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/total_num, 100.*correct/total_num, correct, total_num))
        
    
    # print validation error
    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('Validation Error: ', correct / total_num) 
    
    # schedule learning rate decay
    optimizer=exp_lr_scheduler(epoch, optimizer, strategy=args.lr_schedule, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)

torch.save(model.state_dict(), args.name + '_result/'+args.arch+'_baseline'+'.pkl')  
