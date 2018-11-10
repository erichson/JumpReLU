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
parser.add_argument('--name', type=str, default='cifar100', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
#
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='learning rate (default: 0.01)')
#
parser.add_argument('--lr-decay', type=float, default=0.2, help='learning rate ratio')
#
parser.add_argument('--lr-schedule', type=str, default='normal', help='learning rate schedule')
#
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[30,60], help='Decrease learning rate at these epochs.')
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
parser.add_argument('--resume', type=str, default='cifar10_result/Net_CIFARbaseline1.pkl',  help='choose the resume')
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
    'JumpNet_EMNIST': JumpNet_EMNIST(),
    'Net_CIFAR': Net_CIFAR()
}


model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(args.resume))

#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')

output = []

for param in model.parameters():
    output.append(param.data.cpu().numpy())
    
    

