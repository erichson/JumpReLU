from __future__ import print_function
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad

import scipy.io as lmd
from attack_method import *
#from progressbar import *
from wide_resnet import *
from utils import *
from model import *

import os

from advfuns import *



#==============================================================================
# Attack settings
#==============================================================================
parser = argparse.ArgumentParser(description='Attack Example')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--norm', type=int, default=2, metavar='S', help='2')

parser.add_argument('--classes', type=int, default=9, metavar='S', help='')

parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

parser.add_argument('--eps', type=float, default=0.01, metavar='E', help='how far to perturb input in the negative gradient sign')

parser.add_argument('--arch', type=str, default='Net', help='choose an architecture')

parser.add_argument('--resume', type=str, default='net.pkl', help='choose an existing model')

parser.add_argument('--dataset', type=str, default='mnist', help='chose dataset')

parser.add_argument('--data-set', type=str, default='test', help='chose dataset')

parser.add_argument('--second-order-attack', type=int, default=0, help='second order attack')

parser.add_argument('--iter', type=int, default=100, help='number of iterations')

parser.add_argument('--shift', type=float, default=0.0, metavar='E', help='shift value')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)



if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader, test_loader = getData(name=args.dataset, train_bs=args.test_batch_size, test_bs=args.test_batch_size)
bz = args.test_batch_size





for arg in vars(args):
    print(arg, getattr(args, arg))

#==============================================================================
# get model 
#==============================================================================
model_dct = {'0': Net_CIFAR(shift=0.0),
             '1': Net_CIFAR(shift=0.1),
             '2': Net_CIFAR(shift=0.2),
             '3': Net_CIFAR(shift=0.3),
             '4': Net_CIFAR(shift=0.4),
             '5': Net_CIFAR(shift=0.5),
             '6': Net_CIFAR(shift=0.6)}

for m in model_dct:

    model_dct[m] = torch.nn.DataParallel(model_dct[m].cuda())
    model_dct[m].load_state_dict(torch.load(args.resume))
    model_dct[m].eval()

#==============================================================================
# Begin attack
#==============================================================================

stat_time = time.time()
for m in model_dct:
    num_data = 10000
    X_ori = torch.Tensor(num_data, 3, 32, 32)
    X_fgsm = torch.Tensor(num_data, 3, 32, 32)
    Y_test = torch.LongTensor(num_data)
    
    for i, (data, target) in enumerate(test_loader):

        X_ori[i*bz:(i+1)*bz, :] = data
        Y_test[i*bz:(i+1)*bz] = target
        
        X_fgsm[i*bz:(i+1)*bz,:], a = fgsm_adaptive_iter(model_dct[m], data, target, args.eps, iter=args.iter)
    
    acc = []
    for mt in model_dct:
        result_acc, result_ent = test(X_fgsm, Y_test, model_dct[mt], args)
        acc.append(result_acc)
        

    print('Accuracy for model: ', m, acc)
