import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad

import time
import numpy as np
import scipy

from copy import deepcopy


def generate_class(n, target):
    Y_adv = target * 0 
    for i in range(n):
        Y_adv[i:i+1] = torch.randperm(10)[0]
        while Y_adv[i:i+1].numpy() == target[i:i+1].numpy():
            Y_adv[i:i+1] = torch.randperm(10)[0]
    return Y_adv


#################################################
## FGSM
#################################################
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

def fgsm_v2(model, coder, data, target, eps, two_path=False):
    """Generate an adversarial pertubation using the fast gradient sign method.

    Args:
        data: input image to perturb
    """
    model.eval()
    coder.eval()
    data, target = data.cuda(), target.cuda()
    data.requires_grad = True
    model.zero_grad()
    coder.zero_grad()
    if two_path == False:
        output = model(coder(data))
        loss = F.cross_entropy(output, target)
        loss.backward(create_graph=False)
    else:
        recon_output = coder(data)
        ori_output = model(data)
        recon_loss = F.cross_entropy(recon_output, target)
        ori_loss = F.cross_entropy(ori_output, target)
        loss = recon_loss + ori_loss
        loss.backward(create_graph=False)

    pertubation = eps * torch.sign(data.grad.data)
    x_fgsm = data.data + pertubation
    X_adv = torch.clamp(x_fgsm, torch.min(data.data), torch.max(data.data))

    return X_adv

def fgm(model, data, target, eps):
    """Generate an adversarial pertubation using the fast gradient method.

    Args:
        data: input image to perturb
    """
    bs = data.size(0)
    model.eval()
    data, target = Variable(data.cuda(), requires_grad=True), target.cuda()
    #data.requires_grad = True
    model.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward(create_graph=False)
    pertubation = data.grad.data.view(bs, -1)
    pertubation = pertubation / (torch.norm(pertubation, p=2, dim=1).view(bs,1)+1e-6)
    x_fgsm = data.data + eps * pertubation.view(data.size())
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

def fgsm_adaptive_iter(model, data_, target, eps, iter):
    data = deepcopy(data_)
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
    return data.data



#################################################
## Select Attack Index 
#################################################
def select_index(model, data, c=9, p=2, worst_case = False):
    """Generate an adversarial pertubation using the dp method.

    Args:
        data: input image to perturb
    """
    model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    #output = F.softmax(output) 
    output, ind = torch.sort(output, descending=True)
    n = len(data)

    true_out = output[range(len(data)), n*[0]]
    z_true = torch.sum(true_out)
    data.grad = None
    z_true.backward(retain_graph=True)
    true_grad = data.grad
    pers = torch.zeros(len(data), 1+c).cuda()
    for i in range(1,1+c):
        z = torch.sum(output[:,i])
        data.grad = None
        model.zero_grad()
        z.backward(retain_graph=True)
        grad = data.grad # batch_size x 3k
        grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=p,dim=1) # batch_size x 1
        pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
    if not worst_case:
        pers[range(n),n*[0]] = np.inf
        pers[pers < 0] = 0
        per, index = torch.min(pers,1) # batch_size x 1
    else:
        pers[range(n),n*[0]] = -np.inf
        per, index = torch.max(pers,1) # batch_size x 1
    
    output = []
    for i in range(data.size(0)):
        output.append(ind[i, index[i]].item())
    return torch.LongTensor(output) 

#################################################
## Deep Fool
#################################################

def deep_fool(model, data,target, target_ind, p=2):
    """Generate an adversarial pertubation using the dp method.

    Args:
        data: input image to perturb
    """
    model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    #output = F.softmax(output) 
    n = len(data)

    output_g = output[range(n), target_ind] - output[range(n), target]
    z = torch.sum(output_g)

    data.grad = None
    model.zero_grad()
    z.backward()
    update = data.grad.data + 0.
    update = update.view(n,-1)
    per = (-output_g.data.view(n,-1) + 0.) / (torch.norm(update, p=p, dim=1).view(n,1)+1e-6)
    
    if p == 1:
        update = torch.sign(update)
    elif p ==2:
        update = update.view(n,-1)
        update = update / (torch.norm(update, p=2, dim=1).view(n,1)+1e-6)
    X_adv = data.data + (((per+1e-4)*1.02).view(n,-1)*update.view(n,-1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))
    return X_adv

def deep_fool_iter(model, data, target,c=9, p=2, iter=10, worst_case = False):
    X_adv = data.cuda() + 0.0
    target_ind = select_index(model, data, c=c,p=p, worst_case = worst_case) 

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
#         print(attack_mask)
#         print(target_ind)
#         print(target_ind[attack_mask])
        X_adv[attack_mask,:] = deep_fool(model, X_adv[attack_mask,:],target[attack_mask], target_ind[attack_mask], p=p)
    return X_adv #, update_num

#################################################
## TR First Order Attack
#################################################
def tr_attack(model, data, true_ind, target_ind, eps, p = 2):
    """Generate an adversarial pertubation using the TR method.
    Pick the top false label and perturb towards that.
    First-order attack

    Args:
        data: input image to perturb
        true_ind: is true label
        target_ind: is the attack label
    """
    model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    n = len(data)

    q = 2
    if p == 8:
        q = 1
    
    output_g = output[range(n), target_ind] - output[range(n), true_ind]
    z = torch.sum(output_g)

    data.grad = None
    model.zero_grad()
    z.backward()
    update = deepcopy(data.grad.data) 
    update = update.view(n,-1)
    per = (-output_g.data.view(n,-1) + 0.) / (torch.norm(update, p = q, dim = 1).view(n, 1) + 1e-6)

    if p == 8 or p == 1:
        update = torch.sign(update)
    elif p ==2:
        update = update.view(n, -1)
        update = update / (torch.norm(update, p = 2, dim = 1).view(n,1) + 1e-6)
    per = per.view(-1)
    per_mask = per > eps
    per_mask = per_mask.nonzero().view(-1)
    # set overshoot for small pers
    per[per_mask] = eps
    X_adv = data.data + (((per + 1e-4) * 1.02).view(n,-1) * update.view(n, -1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))
    return X_adv

            
def tr_attack_iter(model, data, target, eps, c = 9, p = 2, iter = 100, worst_case = False):
    X_adv = deepcopy(data.cuda()) 
    target_ind = select_index(model, data, c = c,p = p, worst_case = worst_case) 
    
    update_num = 0.
    for i in range(iter):
        model.eval()
        Xdata, Ytarget = X_adv, target.cuda()
        # First check if the input is correctly classfied before attack
        Xoutput = model(Xdata)
        Xpred = Xoutput.data.max(1, keepdim = True)[1] # get the index of the max log-probability
        tmp_mask = Xpred.view_as(Ytarget) == Ytarget.data # get index
        update_num += torch.sum(tmp_mask.long())
         # if all images are incorrectly classfied the attack is successful and exit
        if torch.sum(tmp_mask.long()) < 1:
            return X_adv#.cpu()#, update_num      
        attack_mask = tmp_mask.nonzero().view(-1)
        X_adv[attack_mask,:]  = tr_attack(model, X_adv[attack_mask,:], target[attack_mask], target_ind[attack_mask], eps, p = p)
    return X_adv#.cpu(), update_num      


#################################################
## TR First Order Attack Adaptive
#################################################
def tr_attack_adaptive(model, data, true_ind, target_ind, eps, p = 2):
    """Generate an adversarial pertubation using the TR method with adaptive
    trust radius.
    Args:
        data: input image to perturb
        true_ind: is true label
        target_ind: is the attack label
    """
    model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    n = len(data)

    q = 2
    if p == 8:
        q = 1

    output_g = output[range(n), target_ind] - output[range(n), true_ind]
    z = torch.sum(output_g)

    data.grad = None
    model.zero_grad()
    z.backward()
    update = deepcopy(data.grad.data) 
    update = update.view(n,-1)
    per = (-output_g.data.view(n, -1) + 0.) / (torch.norm(update, p = q, dim = 1).view(n, 1) + 1e-6)
    
    if p == 8:
        update = torch.sign(update)
    elif p ==2:
        update = update.view(n, -1)
        update = update / (torch.norm(update, p = 2, dim = 1).view(n, 1) + 1e-6)
    
    ### set large per to eps
    per = per.view(-1)
    eps = eps.view(-1)
    per_mask = per > eps
    per_mask = per_mask.nonzero().view(-1)
    # set overshoot for small pers
    per[per_mask] = eps[per_mask]
    per = per.view(n, -1)
    eps = deepcopy(per)
    X_adv = data.data + (1.02 * (eps + 1e-4) * update.view(n, -1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))

    ### update eps magnitude
    ori_diff = -output_g.data + 0.0 

    adv_output = model(X_adv)    
    adv_diff = adv_output[range(n), true_ind] - output[range(n), target_ind]

    eps = eps.view(-1)
    obj_diff = (ori_diff - adv_diff) / eps 

    increase_ind = obj_diff > 0.9 
    increase_ind = increase_ind.nonzero().view(-1)

    decrease_ind = obj_diff < 0.5
    decrease_ind = decrease_ind.nonzero().view(-1)

    eps[increase_ind] = eps[increase_ind] * 1.2
    eps[decrease_ind] = eps[decrease_ind] / 1.2

    if p == 2:
        eps_max = 0.05
        eps_min = 0.0005
        eps_mask = eps > eps_max
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_max
        eps_mask = eps < eps_min
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_min

    elif p == 8 or p==1:
        eps_max = 0.01
        eps_min = 0.0001
        eps_mask = eps > eps_max
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_max
        eps_mask = eps < eps_min
        eps_mask = eps_mask.nonzero().view(-1)
        eps[eps_mask] = eps_min

    eps = eps.view(n, -1)
    return X_adv, eps

def tr_attack_adaptive_iter(model, data, target, eps, c = 9, p = 2, iter = 100, worst_case = False):
    X_adv = deepcopy(data.cuda())
    target_ind = select_index(model, data, c=c,p=p, worst_case = worst_case) 
    
    update_num = 0.
    eps = torch.from_numpy(np.array([eps] * len(data))).view(len(data), -1)
    eps = eps.type(torch.FloatTensor).cuda()
    for i in range(iter):
        model.eval()
        Xdata, Ytarget = X_adv, target.cuda()
        Xoutput = model(Xdata)
        Xpred = Xoutput.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        tmp_mask = Xpred.view_as(Ytarget) == Ytarget.data # get index
        update_num += torch.sum(tmp_mask.long())
        if torch.sum(tmp_mask.long()) < 1:
            return X_adv.cpu(), update_num      
        attack_mask = tmp_mask.nonzero().view(-1)
        X_adv[attack_mask,:], eps[attack_mask,:]  = tr_attack_adaptive(model, X_adv[attack_mask,:], target[attack_mask], target_ind[attack_mask], eps[attack_mask,:], p = p)
    return X_adv.cpu(), update_num  

#################################################
## JSMA
#################################################
def jsma(model, data, targets, epochs=1, eps=1.0, k=1, clip_min=0.0, clip_max=1.0):
    target = torch.zeros(targets.size()).long().cuda()
    for i in range(data.size()[0]):
        while (True):
            target[i] = torch.randint(0, 10, (1,))
            if target[i] != targets[i]:
                break
    model.eval()
    x_adv = data.clone()
    for i in range(epochs):
        x_adv, target = Variable(x_adv.cuda(), requires_grad=True), target.cuda()
        model.zero_grad()
        output = model(x_adv)
        dy = F.softmax(output, dim=1)
        #Compute gradient of loss fn w.r.t. the inputs
        model.zero_grad()
        da = torch.sum(dy)
        da.backward(retain_graph=True)
        dy_dx = x_adv.grad.data.clone()
        
        #Compute gradient of targets w.r.t. the inputs
        model.zero_grad()
        dt = torch.sum(dy.gather(1, target.view(-1, 1)))
        dt.backward()
        dt_dx = x_adv.grad.data.clone()
        
        #Compute gradient of non-targets w.r.t. the inputs
        do_dx = dy_dx - dt_dx
        
        #Compute all the conditions
        c0 = 1 if eps < 0 else x_adv < clip_max
        c1 = 1 if eps > 0 else x_adv > clip_min
        c2 = dt_dx >= 0
        c3 = do_dx <= 0
        cond = (c0 * c1 * c2 * c3).float()
        
        # saliency score for each pixel
        score = cond * dt_dx * do_dx.abs()
        score = score.view(score.size()[0], -1)
        _, idx = score.max(k)
                
        #Update the adversarial inputs
        x_adv = x_adv.view(x_adv.size()[0], -1)
        #TODO: Need to optimize
        for j in range(x_adv.size()[0]):
            x_adv.data[j, idx[j]] += eps
        #x_adv = x_adv.view(x_adv.size()[0], 3, 32, 32)
        x_adv = x_adv.view(data.data.size())
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
        x_adv.grad = None
        #print(np.sum(np.abs((x_adv-data).detach().cpu().numpy()) > 0))
    return x_adv
