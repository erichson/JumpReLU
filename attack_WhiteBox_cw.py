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
from utils import *



import os

from advfuns import *
from models import *


from prettytable import PrettyTable
import pandas as pd

import foolbox as fb

#==============================================================================
# Attack settings
#==============================================================================
parser = argparse.ArgumentParser(description='Attack Example for MNIST')

parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N', help='input batch size for testing (default: 1000)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--eps', type=float, default=0.01, metavar='E', help='how far to perturb input in the negative gradient sign')

parser.add_argument('--arch', type=str, default='LeNetLike', help='choose an architecture')

parser.add_argument('--resume', type=str, default='mnist_result/LeNetLike_baseline.pkl', help='choose an existing model')

parser.add_argument('--dataset', type=str, default='mnist', help='chose dataset')

parser.add_argument('--iter', type=int, default=40, help='number of iterations for FGSM')

parser.add_argument('--iter_df', type=int, default=40, help='number of iterations for DeepFool')

parser.add_argument('--jump', type=float, nargs='+', default=[0.0], help='jump value')

parser.add_argument('--runs', type=int, default=1, help='number of simulations')

parser.add_argument('--depth', type=int, default=20, help='choose the depth of resnet')
#

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)



if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader, test_loader = getData(name=args.dataset, train_bs=args.test_batch_size, test_bs=args.test_batch_size)
batchSize = args.test_batch_size



for arg in vars(args):
    print(arg, getattr(args, arg))


if not os.path.exists('results'):
    os.makedirs('results')

#================================================
# begin simulation
#================================================
accuracy = pd.DataFrame()
relative = pd.DataFrame()
absolute = pd.DataFrame()
attack_time = pd.DataFrame()




for jump in args.jump:
    
    
    #================================================
    # get model
    #================================================
    model_list = {
            'LeNetLike': LeNetLike(jump = jump),
            'AlexLike': AlexLike(jump = jump),
            'JumpResNet': JumpResNet(depth=args.depth, jump = jump),
            'MobileNetV2': MobileNetV2(jump = jump),      
    }
    
    
    model = model_list[args.arch]
    if args.cuda:
        model.cuda()
    
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    
    
    
    #==============================================================================
    # Begin attack
    #==============================================================================
    for irun in range(args.runs):
        if args.dataset == 'mnist':
            num_data = 10000
            num_class = 9

            X_ori = torch.Tensor(num_data, 1, 28, 28)
            X_fgsm = torch.Tensor(num_data, 1, 28, 28)
            CLIP_MIN = 0.4243
            CLIP_MAX = 2.8215 
        elif args.dataset == 'cifar10':
            num_data = 100
            num_class = 9
            
            X_ori = torch.Tensor(num_data, 3, 32, 32)
            X_fgsm = torch.Tensor(num_data, 3, 32, 32)
            CLIP_MIN = -2.4291
            CLIP_MAX = 2.7538 

        
        fb_model = fb.models.PyTorchModel(model, bounds=(CLIP_MIN,CLIP_MAX), num_classes=10)
        cw_attack = fb.attacks.CarliniWagnerL2Attack(fb_model)
        iter_fgsm = 0.
        iter_dp1 = 0.
        iter_dp2 = 0.
        
        
        Y_test = torch.LongTensor(num_data)
        
        
        
        
        
        print('Run IFGSM')
        stat_time = time.time()
        for i, (data, target) in enumerate(test_loader):
        
            X_ori[i*batchSize:(i+1)*batchSize, :] = data
            Y_test[i*batchSize:(i+1)*batchSize] = target
            
            X_fgsm[i*batchSize:(i+1)*batchSize,:] = torch.from_numpy(cw_attack(data.numpy()[0,:], target.numpy()[0], learning_rate=0.01))

            if i == num_data:
                break
        #print('iters: ', iter_fgsm)
        time_ifgsm = time.time() - stat_time
        print('total_time: ', time_ifgsm)
            
        
        
        
        
        result_acc = np.zeros(7)
        result_ent = np.zeros(7)
        result_dis = np.zeros(7)
        result_dis_abs = np.zeros(7)
        result_large = np.zeros(7)
        
        
        result_acc[0], result_ent[0] = test_ori(model, test_loader, num_data, args)
        result_acc[1], result_ent[1] = test_adv(X_fgsm, Y_test, model, num_data, args)
        
        
        # FGSM inf norm
        result_dis[1], result_dis_abs[1],  result_large[1]= distance(X_fgsm, X_ori, norm=1)
        
        # FGSM two norm
        result_dis[2], result_dis_abs[2],  result_large[2]= distance(X_fgsm, X_ori, norm=2)
        
        #***********************
        # Print results
        #***********************
        print(result_acc)
        print(result_dis)
        print(result_dis_abs)
        
        
