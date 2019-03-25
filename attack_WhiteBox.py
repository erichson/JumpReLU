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


#==============================================================================
# Attack settings
#==============================================================================
parser = argparse.ArgumentParser(description='Attack Example for MNIST')

parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N', help='input batch size for testing (default: 1000)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--eps', type=float, default=0.01, metavar='E', help='how far to perturb input in the negative gradient sign')

parser.add_argument('--eps_tr', type=float, default=0.01, metavar='ET', help='how far to perturb input with trust region direction')

parser.add_argument('--arch', type=str, default='LeNetLike', help='choose an architecture')

parser.add_argument('--resume', type=str, default='mnist_result/LeNetLike_baseline.pkl', help='choose an existing model')

parser.add_argument('--dataset', type=str, default='mnist', help='chose dataset')

parser.add_argument('--iter', type=int, default=40, help='number of iterations for FGSM')

parser.add_argument('--iter_df', type=int, default=40, help='number of iterations for DeepFool')

parser.add_argument('--iter_tr', type=int, default=40, help='number of iterations for TrustRegion Attack')

parser.add_argument('--jump', type=float, nargs='+', default=[0.0], help='jump value')

parser.add_argument('--runs', type=int, default=1, help='number of simulations')

parser.add_argument('--depth', type=int, default=34, help='choose the depth of resnet')
#
#
parser.add_argument('--id', type=str, default='det', help='identifier for saving results')
#
parser.add_argument('--widen_factor', type=int, default=4, metavar='E', help='Widen factor')

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
            'ResNet': ResNet(depth=20, jump = jump),
            'MobileNetV2': MobileNetV2(jump = jump),     
            'WideResNet': WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropout_rate=0.3, num_classes=10, level=1, jump=jump),          
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
            X_deepfool1 = torch.Tensor(num_data, 1, 28, 28)
            X_deepfool2 = torch.Tensor(num_data, 1, 28, 28)            
            X_tr = torch.Tensor(num_data, 1, 28, 28)            
            
        elif args.dataset == 'cifar10':
            num_data = 10000
            num_class = 9
            
            X_ori = torch.Tensor(num_data, 3, 32, 32)
            X_fgsm = torch.Tensor(num_data, 3, 32, 32)
            X_deepfool1 = torch.Tensor(num_data, 3, 32, 32)
            X_deepfool2 = torch.Tensor(num_data, 3, 32, 32)            
            X_tr = torch.Tensor(num_data, 3, 32, 32)            

        
        iter_fgsm = 0.
        iter_dp1 = 0.
        iter_dp2 = 0.
        iter_tr = 0.
        
        
        Y_test = torch.LongTensor(num_data)
        
        
        
        
        
        print('Run IFGSM')
        stat_time = time.time()
        for i, (data, target) in enumerate(test_loader):
        
            X_ori[i*batchSize:(i+1)*batchSize, :] = data
            Y_test[i*batchSize:(i+1)*batchSize] = target
            
            X_fgsm[i*batchSize:(i+1)*batchSize,:], a = fgsm_adaptive_iter(model, data, target, args.eps, iterations=args.iter)
            #iter_fgsm += a
        #print('iters: ', iter_fgsm)
        time_ifgsm = time.time() - stat_time
        print('total_time: ', time_ifgsm)
            
        
        print('Run DeepFool (inf norm)')
        stat_time = time.time()
        for i, (data, target) in enumerate(test_loader):
            X_deepfool1[i*batchSize:(i+1)*batchSize,:], a = deep_fool_iter(model, data, target, c=num_class, p=1, iterations=args.iter_df)
            iter_dp1 += a
        print('iters: ', iter_dp1)
        time_deepfool_inf = time.time() - stat_time
        print('total_time: ', time_deepfool_inf)
        
        
        print('Run DeepFool (two norm)')
        stat_time = time.time()
        for i, (data, target) in enumerate(test_loader):
            X_deepfool2[i*batchSize:(i+1)*batchSize,:], a = deep_fool_iter(model, data, target, c=num_class, p=2, iterations=args.iter_df)
            iter_dp2 += a
        print('iters: ', iter_dp2)
        time_deepfool_two = time.time() - stat_time        
        print('total_time: ', time_deepfool_two)
        
        
        print('Run Trust Region Attack')
        stat_time = time.time()
        for i, (data, target) in enumerate(test_loader):
            X_tr[i*batchSize:(i+1)*batchSize,:], a = tr_attack_iter(model, data, target, args.eps_tr, c=num_class, p=2, iterations=args.iter_tr)
            iter_tr += a
        print('iters: ', iter_tr)
        time_tr = time.time() - stat_time        
        print('total_time: ', time_tr)
        

        result_acc = np.zeros(9)
        result_ent = np.zeros(9)
        result_dis = np.zeros(9)
        result_dis_abs = np.zeros(9)
        result_large = np.zeros(9)
        
        
        result_acc[0], result_ent[0] = test_ori(model, test_loader, num_data, args)
        result_acc[1], result_ent[1] = test_adv(X_fgsm, Y_test, model, num_data, args)
        result_acc[2], result_ent[2] = test_adv(X_deepfool1, Y_test, model, num_data, args)
        result_acc[3], result_ent[3] = test_adv(X_deepfool2, Y_test, model, num_data, args)
        result_acc[4], result_ent[4] = test_adv(X_tr, Y_test, model, num_data, args)
        
        
        # FGSM inf norm
        result_dis[1], result_dis_abs[1],  result_large[1]= distance(X_fgsm, X_ori, norm=1)
        
        # FGSM two norm
        result_dis[2], result_dis_abs[2],  result_large[2]= distance(X_fgsm, X_ori, norm=2)
        
        # Deepfool (inf) inf norm
        result_dis[3], result_dis_abs[3],  result_large[3]= distance(X_deepfool1, X_ori, norm=1)

        # Deepfool (inf) two norm
        result_dis[4], result_dis_abs[4],  result_large[4]= distance(X_deepfool1, X_ori, norm=2)
                
        # Deepfool (two) inf norm
        result_dis[5], result_dis_abs[5],  result_large[5]= distance(X_deepfool2, X_ori, norm=1)
        
        # Deepfool (two) two norm
        result_dis[6], result_dis_abs[6],  result_large[6]= distance(X_deepfool2, X_ori, norm=2)        
        
        # TR inf norm
        result_dis[7], result_dis_abs[7],  result_large[7]= distance(X_tr, X_ori, norm=1)
        
        # TR two norm
        result_dis[8], result_dis_abs[8],  result_large[8]= distance(X_tr, X_ori, norm=2)        
        
        #***********************
        # Print results
        #***********************
        print('Jump value: ', jump)
        x = PrettyTable()
        x.field_names = [" ", "Clean Data", "IFGSM", "DeepFool_inf", "DeepFool", "TR"]
        x.add_row(np.hstack(('Accuracy: ',   np.round(result_acc[([0,1,2,3,4])], 5))))
        x.add_row(np.hstack(('Rel. Noise: ', np.round(result_dis[([0,1,3,6,8])], 5))))
        x.add_row(np.hstack(('Abs. Noise: ', np.round(result_dis_abs[([0,1,3,6,8])], 5))))
        print(x)
        
        
        #***********************
        # Add to pandas df
        #***********************
        
        
        s = pd.Series({"jump" : jump, 
                       "clean": np.round(result_acc[0], 3) , 
                       "IFGSM_inf": np.round(result_acc[1], 3),
                       "IFGSM_two": np.round(result_acc[1], 3), 

                       "DeepFool_max_inf": np.round(result_acc[2], 3), 
                       "DeepFool_max_two": np.round(result_acc[2], 3),

                       "DeepFool_inf": np.round(result_acc[3], 3), 
                       "DeepFool_two": np.round(result_acc[3], 3)                       
                       })    
        accuracy = accuracy.append(s, ignore_index=True)    
        
        
        s = pd.Series({"jump" : jump, 
                       "clean": np.round(result_dis[0], 3) , 
                       "IFGSM_inf": np.round(result_dis[1], 3),
                       "IFGSM_two": np.round(result_dis[2], 3), 

                       "DeepFool_max_inf": np.round(result_dis[3], 3), 
                       "DeepFool_max_two": np.round(result_dis[4], 3),

                       "DeepFool_inf": np.round(result_dis[5], 3), 
                       "DeepFool_two": np.round(result_dis[6], 3)                       
                       }) 
        relative = relative.append(s, ignore_index=True)    
            
            
        s = pd.Series({"jump" : jump, 
                       "clean": np.round(result_dis_abs[0], 3) , 
                       "IFGSM_inf": np.round(result_dis_abs[1], 3),
                       "IFGSM_two": np.round(result_dis_abs[2], 3), 

                       "DeepFool_max_inf": np.round(result_dis_abs[3], 3), 
                       "DeepFool_max_two": np.round(result_dis_abs[4], 3),

                       "DeepFool_inf": np.round(result_dis_abs[5], 3), 
                       "DeepFool_two": np.round(result_dis_abs[6], 3)                       
                       }) 
        absolute = absolute.append(s, ignore_index=True)    


        s = pd.Series({"jump": jump, 
                       "IFGSM": time_ifgsm, 
                       "DeepFool_inf": time_deepfool_inf, 
                       "DeepFool_two": time_deepfool_two})
        attack_time = attack_time.append(s, ignore_index=True)    

        accuracy.to_pickle("results/" + args.arch + "_" + "accuracy" + "_" + args.id + ".pkl")
        relative.to_pickle("results/" + args.arch + "_" + "relative" + "_" + args.id + ".pkl")
        absolute.to_pickle("results/" + args.arch + "_" + "absolute" + "_" + args.id + ".pkl")
        attack_time.to_pickle("results/" + args.arch + "_" + "time" + "_" + args.id + ".pkl")
            
