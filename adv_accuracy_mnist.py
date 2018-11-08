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
model_list = {
    'JumpNet': JumpNet(shift=args.shift),
    'JumpNet_EMNIST': JumpNet_EMNIST(shift=args.shift)
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

stat_time = time.time()
if args.data_set == 'test':
    if args.dataset == 'mnist':
        num_data = 10000
    else:
        num_data = 18800
    X_ori = torch.Tensor(num_data, 1, 28, 28)
    X_fgsm = torch.Tensor(num_data, 1, 28, 28)
    X_deepfool1 = torch.Tensor(num_data, 1, 28, 28)
    X_deepfool2 = torch.Tensor(num_data, 1, 28, 28)

    iter_fgsm = 0.
    iter_dp1 = 0.
    iter_dp2 = 0.

    Y_test = torch.LongTensor(num_data)
    
    for i, (data, target) in enumerate(test_loader):

        X_ori[i*bz:(i+1)*bz, :] = data
        Y_test[i*bz:(i+1)*bz] = target
        
        X_fgsm[i*bz:(i+1)*bz,:], a = fgsm_adaptive_iter(model, data, target, args.eps, iter=args.iter)
        iter_fgsm += a
        
        X_deepfool1[i*bz:(i+1)*bz,:], a = deep_fool_iter(model, data, target,c=args.classes, p=1, iter=args.iter)
        iter_dp1 += a
        
        X_deepfool2[i*bz:(i+1)*bz,:], a = deep_fool_iter(model, data, target,c=args.classes, p=2, iter=args.iter)
        iter_dp2 += a

        print('current iteration: ', i)


    if not os.path.exists('generate_data'):
        os.makedirs('generate_data')

#     torch.save([X_ori, X_fgsm, X_deepfool, X_tr_first, X_tr_first_adp, X_tr_second, Y_test], 'generate_data/'+args.arch.lower()+str(args.norm)+'.pt',)    


print('iters: ', iter_fgsm, iter_dp1, iter_dp2)
print('total_time: ', time.time()-stat_time)



result_acc = np.zeros(4)
result_ent = np.zeros(4)
result_dis = np.zeros(4)
result_large = np.zeros(4)

result_acc[0], result_ent[0] = test_ori(model, test_loader, args)
result_acc[1], result_ent[1] = test(X_fgsm, Y_test, model, args)
result_acc[2], result_ent[2] = test(X_deepfool1, Y_test, model, args)
result_acc[3], result_ent[3] = test(X_deepfool2, Y_test, model, args)


result_dis[1],result_large[1]= distance(X_fgsm,X_ori, norm=1)
result_dis[2],result_large[2]= distance(X_deepfool1,X_ori, norm=1)
result_dis[3],result_large[3]= distance(X_deepfool2,X_ori, norm=2)


print('Accuracy: ', np.round(result_acc, 4))
#print(result_ent)
print('Noise Lev:', np.round(result_dis, 4))
#print(result_large)
