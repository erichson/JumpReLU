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
from advfuns import *
from utils import *
from models import *
import os

#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--name', type=str, default='cifar10', metavar='N', help='dataset')
#
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
#
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.01)')
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
parser.add_argument('--arch', type=str, default='JumpResNet',  help='choose the architecture')
#
parser.add_argument('--large-ratio', type=int, default=1, metavar='N',  help='large ratio')
#
parser.add_argument('--depth', type=int, default=20, help='choose the depth of resnet')
#
parser.add_argument('--jump', type=float, default=0.0, metavar='E', help='Jump value')
#
parser.add_argument('--resume', type=str, default='net.pkl', help='choose an existing model')
#
parser.add_argument('--temp', '--tp', default=100, type=float, metavar='T', help='')

#
args = parser.parse_args()

args.cuda=True

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

# for distillation

if args.arch == 'LeNetLike':
    model1 = torch.nn.DataParallel(LeNetLike(args.jump)).cuda()
    model1.load_state_dict(torch.load(args.resume))
    model1.eval()

    model2 = torch.nn.DataParallel(LeNetLike(args.jump)).cuda()
elif args.arch == 'AlexLike':
    model1 = torch.nn.DataParallel(AlexLike(args.jump)).cuda()
    model1.load_state_dict(torch.load(args.resume))
    model1.eval()

    model2 = torch.nn.DataParallel(AlexLike(args.jump)).cuda()
elif args.arch == 'JumpResNet':
    model1 = torch.nn.DataParallel(JumpResNet(depth=args.depth, jump=args.jump)).cuda()
    model1.load_state_dict(torch.load(args.resume))
    model1.eval()

    model2 = torch.nn.DataParallel(JumpResNet(depth=args.depth, jump=args.jump)).cuda()
elif args.arch == 'MobileNetV2':
    model1 = torch.nn.DataParallel(MobileNetV2(args.jump)).cuda()
    model1.load_state_dict(torch.load(args.resume))
    model1.eval()

    model2 = torch.nn.DataParallel(MobileNetV2(args.jump)).cuda()
else:
    raise('This arch is not supported now.')

#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model1.parameters())/1000000.0))
print('************')

#print(model2)

#==============================================================================
# Run
#==============================================================================
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

inner_loop = 0
num_updates = 0

Temp = args.temp
#Fs = F.softmax()

for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.size()[0] < args.batch_size:
            continue
        model2.train()
        # gather input and target for large batch training        
        inner_loop += 1
        # get small model update
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output1 = model1(data) / Temp
            output1 = F.softmax(output1, dim=1)

        output2 = model2(data) / Temp
        output2 = F.softmax(output2, dim=1)
     
        optimizer.zero_grad()
        loss = -torch.sum(output1 * torch.log(output2+1e-6))/len(target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*target.size()[0]
        total_num += target.size()[0]
        _, predicted = output2.max(1)
        correct += predicted.eq(target).sum().item()
        
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/total_num, 100.*correct/total_num, correct, total_num))
        
    test_ori(model2, test_loader, 10000, args)   
    optimizer=exp_lr_scheduler(epoch, optimizer, strategy=args.lr_schedule, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)

torch.save(model2.state_dict(), args.name + '_result/'+args.arch+'_distillation'+'.pkl')  
