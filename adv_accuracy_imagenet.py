import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from vgg_j import *

from attack_method import *
from data_tools import *



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--eps', default=0.01, type=float,
                     help='')
parser.add_argument('--classes', default=9, type=int,
                    help='')
parser.add_argument('--norm', default=2, type=int,
                    help='')
parser.add_argument('--iter', default=100, type=int,
                    help='GPU id to use.')
parser.add_argument('--worst-case', default=0, type=int,
                    help='use worst-case attack')
parser.add_argument('--shift', default=0.0, type=float,
                     help='')
best_acc1 = 0

def main():
    global args, best_acc1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    model_list = {
    'vgg16': vgg16(pretrained=True, shift=args.shift)
    }
    
    model = model_list[args.arch]

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)


    bz = args.batch_size
    
    
    validate(test_loader, model, criterion)
    
    stat_time = time.time()
    X_ori = torch.Tensor(bz,3,224,224)
    X_fgsm = torch.Tensor(bz,3,224,224)    
    X_deepfool1 = torch.Tensor(bz,3,224,224)
    X_deepfool2 = torch.Tensor(bz,3,224,224)

    iter_fgsm = 0.
    iter_dp1 = 0.
    iter_dp2 = 0.


    Y_test = torch.LongTensor(bz)
   
    result_acc = np.zeros(4)
    result_dis = np.zeros(4)
    result_large = np.zeros(4)
    

    for i, (data, target) in enumerate(test_loader):
        X_ori  = data
        Y_test = target
        X_fgsm, a = fgsm_adaptive_iter(model, data, target, 0.005, iter=args.iter)
        iter_fgsm += a
        
        print('current fgsm ', i )
        X_deepfool1, a = deep_fool_iter(model, data, target,c=args.classes, p=1, iter=args.iter)
        iter_dp1 += a
        
        X_deepfool2, a = deep_fool_iter(model, data, target,c=args.classes, p=2, iter=args.iter)
        iter_dp2 += a
    
        acc1 = validate_all(X_ori,Y_test,model, criterion)
        acc2 = validate_all(X_fgsm,Y_test,model, criterion)
        acc3 = validate_all(X_deepfool1,Y_test,model, criterion)
        acc4 = validate_all(X_deepfool2,Y_test,model, criterion)

       
        result_acc[0] += acc1
        result_acc[1] += acc2
        result_acc[2] += acc3
        result_acc[3] += acc4

        dis1, ldis1 = distance(X_fgsm,X_ori, norm=1)
        dis2, ldis2 = distance(X_deepfool1,X_ori,norm=1)
        dis3, ldis3 = distance(X_deepfool2,X_ori,norm=2)

        result_dis[1] += dis1
        result_dis[2] += dis2
        result_dis[3] += dis3

    print(iter_fgsm, iter_dp1, iter_dp2)
    print(result_acc*bz/50000.)
    print(result_dis*bz/50000.)
    print(result_large)


def distance(X_adv, X_prev, norm=2):
    n = len(X_adv)
    dis = 0.
    large_dis = 0.
    for i in range(n):
        if norm == 2:
            tmp_dis = torch.norm(X_adv[i,:].cpu()-X_prev[i,:].cpu(),p=norm)/torch.norm(X_prev[i,:].cpu(), p=norm)
        if norm == 1:
            tmp_dis = torch.max(torch.abs(X_adv[i,:].cpu()-X_prev[i,:].cpu()))/torch.max(torch.abs(X_prev[i,:].cpu()))
        dis += tmp_dis
        large_dis = max(large_dis, tmp_dis)
    return dis/n, large_dis


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate_all(X, Y, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_data = X.size()[0]
    num_iter = num_data//100
    with torch.no_grad():
        end = time.time()
        for i in range(num_iter):
            #if args.gpu is not None:
            input = X[100*i:100*(i+1),:].cuda(args.gpu, non_blocking=True)
            target = Y[100*i:100*(i+1)].cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            #    print('Test: [{0}/{1}]\t'
            #          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #           i, num_iter, batch_time=batch_time, loss=losses,
            #           top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
