from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from progressbar import *
from utils import *
from model import *
import os

#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--name', type=str, default='cifar10', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 64)')
#
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
#
parser.add_argument('--lr-decay', type=float, default=0.2, help='learning rate ratio')
#
parser.add_argument('--lr-schedule', type=str, default='normal', help='learning rate schedule')
#
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[80,120], help='Decrease learning rate at these epochs.')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
#
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
#
parser.add_argument('--arch', type=str, default='ResNet',  help='choose the architecture')
#
parser.add_argument('--large-ratio', type=int, default=1, metavar='N',  help='large ratio')
#
parser.add_argument('--depth', type=int, default=110, help='choose the depth of resnet')
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
    'JumpNet': JumpNet(),
    'JumpNet_EMNIST': JumpNet_EMNIST()
}


model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')



#==============================================================================
# Run
#==============================================================================
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

inner_loop = 0
num_updates = 0

for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size()[0] < args.batch_size:
            continue
        model.train()
        # gather input and target for large batch training        
        inner_loop += 1
        # get small model update
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target) / float(args.large_ratio)
        loss.backward()
        train_loss += loss.item()*target.size()[0]*float(args.large_ratio)
        total_num += target.size()[0]
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        
        if inner_loop % args.large_ratio  == 0:
            num_updates += 1
            optimizer.step()
            inner_loop = 0
            optimizer.zero_grad()
            
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/total_num, 100.*correct/total_num, correct, total_num))
        
    test(model, test_loader)   
    optimizer=exp_lr_scheduler(epoch, optimizer, strategy=args.lr_schedule, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)

torch.save(model.state_dict(), args.name + '_result/'+args.arch+'baseline'+'.pkl')  
