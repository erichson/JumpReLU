import numpy as np
import torch
import torch.nn.functional as F

from utils import *
from model import *




def test_ori(model, test_loader, args):
    model.eval()
    correct = 0
    total_ent = 0.
    total_num = 0.
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            total_num += len(target)
            output = model(data)
            #print(correct)
            ent = F.softmax(output, dim=0)
            tmp_A = sum(ent * torch.log(ent+1e-6))
            total_ent += tmp_A[0]
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()


    print('correct: ', correct / total_num)
    return correct/total_num, total_ent/total_num


def test(adv_data, Y_test, model, args):
    num_data = adv_data.size()[0]
    num_iter = num_data // 100
    model.eval()
    correct = 0
    total_ent = 0.
    with torch.no_grad():
        for i in np.arange(num_iter):
            data, target = adv_data[100*i:100*(i+1), :], Y_test[100*i:100*(i+1)]
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            ent = F.softmax(output, dim=0)
            tmp_A = sum(ent * torch.log(ent+1e-6))
            total_ent += tmp_A[0]
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # print('correct: ', correct / 10000.)
    return correct*1./num_data, total_ent*1./num_data




def distance(X_adv, X_prev, norm):
    n = len(X_adv)
    dis = 0.
    large_dis = 0.
    for i in range(n):
        if norm == 2:
            tmp_dis = torch.norm(X_adv[i,:]-X_prev[i,:],p=2)/torch.norm(X_prev[i,:], p=2)
        if norm == 1:
            tmp_dis = torch.max(torch.abs(X_adv[i,:]-X_prev[i,:]))/torch.max(torch.abs(X_prev[i,:]))
        dis += tmp_dis
        large_dis = max(large_dis, tmp_dis)
    return dis/n, large_dis
