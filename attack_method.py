##############################################################################
## Attacks written by Zhewei Yao <zheweiy@berkeley.edu>
#############################################################################

import torch
import torch.nn.functional as F
from torch.autograd import Variable, grad

import numpy as np


#==============================================================================
## FGSM
#==============================================================================
def fgsm(model, data, target, eps):
    """Generate an adversarial pertubation using the fast gradient sign method.

    Args:
        data: input image to perturb
    """
    model.eval()
    data, target = Variable(data.cuda(), requires_grad=True), target.cuda()
    #data.requires_grad = True
    model.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward(create_graph=False)
    pertubation = eps * torch.sign(data.grad.data)
    x_fgsm = data.data + pertubation
    X_adv = torch.clamp(x_fgsm, torch.min(data.data), torch.max(data.data))

    return X_adv



def fgsm_iter(model, data, target, eps, iter=10):
    """
    iteration version of fgsm
    """
    
    X_adv = fgsm(model, data, target, eps/iter)
    for i in range(iter-1):
    	X_adv = fgsm(model, X_adv, target, eps/iter)
    return X_adv



def fgsm_adaptive_iter(model, data, target, eps, iter):
    update_num = 0
    i = 0
    while True:
        if i >= iter:
            print('failed to fool all the image')
            data = Variable(data)
            break
        model.eval()
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        tmp_mask = pred.view_as(target)==target.data # get index
        update_num += torch.sum(tmp_mask.long())
        if torch.sum(tmp_mask.long()) < 1: # allowed fail
            break
        attack_mask = tmp_mask.nonzero().view(-1)
        data[attack_mask,:] = fgsm(model, data[attack_mask,:], target[attack_mask], eps)
        i += 1
    return data.data, update_num








#==============================================================================
## Deep Fool
#==============================================================================

def deep_fool(model, data, c=9, p=2):
    """Generate an adversarial pertubation using the dp method.

    Args:
        data: input image to perturb
    """
    model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    
    output, ind = torch.sort(output, descending=True)
    #c = output.size()[1]
    n = len(data)

    true_out = output[range(len(data)), n*[0]]
    z_true = torch.sum(true_out)
    data.grad = None
    z_true.backward(retain_graph=True)
    true_grad = data.grad
    grads = torch.zeros([1+c] + list(data.size())).cuda()
    pers = torch.zeros(len(data), 1+c).cuda()
    for i in range(1,1+c):
        z = torch.sum(output[:,i])
        data.grad = None
        model.zero_grad()
        z.backward(retain_graph=True)
        grad = data.grad # batch_size x 3k
        grads[i] = grad.data
        grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=p,dim=1) # batch_size x 1
        pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
    pers[range(n),n*[0]] = np.inf
    pers[pers < 0] = 0
    per, index = torch.min(pers,1) # batch_size x 1
    #print('maximum pert: ', torch.max(per))
    update = grads[index,range(len(data)),:] - true_grad.data
    if p == 1:
        update = torch.sign(update)
    elif p ==2:
        update = update.view(n,-1)
        update = update / (torch.norm(update, p=2, dim=1).view(n,1)+1e-6)
    X_adv = data.data + torch.diag(torch.abs((per+1e-4)*1.02)).mm(update.view(n,-1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))
    return X_adv



def deep_fool_iter(model, data, target,c=9, p=2, iter=10):
    X_adv = data.cuda() + 0.0
    update_num = 0.
    for i in range(iter):
        model.eval()
        Xdata, Xtarget = X_adv, target.cuda()
        Xdata, Xtarget = Variable(Xdata, requires_grad=True), Variable(Xtarget)
        model.zero_grad()
        Xoutput = model(Xdata)
        Xpred = Xoutput.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        tmp_mask = Xpred.view_as(Xtarget)==Xtarget.data # get index
        update_num += torch.sum(tmp_mask.long())
        #print('need to attack: ', torch.sum(tmp_mask))
        if torch.sum(tmp_mask.long()) < 1:
            break
        #print (i, ': ', torch.sum(tmp_mask.long()))
        attack_mask = tmp_mask.nonzero().view(-1)
        X_adv[attack_mask,:] = deep_fool(model, X_adv[attack_mask,:], c=c, p=p)
    return X_adv, update_num
