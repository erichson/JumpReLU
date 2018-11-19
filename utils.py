import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#from progressbar import *



def train(epoch, model, train_loader,optimizer, criterion=nn.CrossEntropyLoss()):
    model.train()
    print('\nTraing, Epoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
                    
def test(model, test_loader):
    print('\nTesting')
    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('testing_correct: ', correct / total_num)
    return correct / total_num


def exp_lr_scheduler(epoch, optimizer, strategy=True, decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    print(strategy)

    if strategy=='normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer



def fb_warmup(optimizer, epoch, baselr, large_ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch * (baselr * large_ratio-baselr) / 5. + baselr
    return optimizer 



# useful
def group_add(params, update, lmbd=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].add_(update[i]*lmbd+0.) 
    return params

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x*y) for (x, y) in zip(xs, ys)])


def get_p_g_m(opt, layers):
    i = 0
    paramlst = []
    grad = []
    mum = []

    for group in opt.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
    
        for p in group['params']:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data 
            d_p = p.grad.data
            if momentum != 0:
                param_state = opt.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
            if i in layers:
                paramlst.append(p.data)
                grad.append(d_p+0.) # get grad
                mum.append(buf*momentum+0.)
            i += 1
    return paramlst, grad, mum


def manually_update(opt, grad):
    for group in opt.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for i,p in enumerate(group['params']):
            d_p = grad[i]
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = opt.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            p.data.add_(-group['lr'], d_p)

def change_lr_single(optimizer, best_lr):
    """change learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = best_lr
    return optimizer
