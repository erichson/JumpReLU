##############################################################################
## Attacks written by Zhewei Yao <zheweiy@berkeley.edu>
#############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad

import time
import numpy as np
import scipy
import cvxpy as cvx
from qcqp import *
import sys


from joblib import Parallel, delayed
import multiprocessing



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





# #==============================================================================
# ## Our First Order Attack
# #==============================================================================
# def xp_attack(model, data, eps, c=9, p=2):
#     """Generate an adversarial pertubation using the xp method.
#     Pick the top false label and perturb towards that.
#     First-order attack

#     Args:
#         data: input image to perturb
#     """
    
#     """Generate an adversarial pertubation using the dp method.

#     Args:
#         data: input image to perturb
#     """
#     model.eval()
#     data = data.cuda()
#     data.requires_grad = True
#     model.zero_grad()
#     output = model(data)
    
#     output, ind = torch.sort(output, descending=True)
#     #c = output.size()[1]
#     n = len(data)

#     true_out = output[range(len(data)), n*[0]]
#     z_true = torch.sum(true_out)
#     data.grad = None
#     z_true.backward(retain_graph=True)
#     true_grad = data.grad+0.
#     grads = torch.zeros([1+c] + list(data.size())).cuda()
#     pers = torch.zeros(len(data), 1+c).cuda()
#     for i in range(1,1+c):
#         z = torch.sum(output[:,i])
#         data.grad = None
#         model.zero_grad()
#         z.backward(retain_graph=True)
#         grad = data.grad+0. # batch_size x 3k
#         grads[i] = grad.data+0.
#         grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=p,dim=1) # batch_size x 1
#         pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
#     pers[range(n),n*[0]] = np.inf
#     pers[pers < 0] = 0
#     per, index = torch.min(pers,1) # batch_size x 1
#     #print('maximum pert: ', torch.max(per))
#     update = grads[index,range(len(data)),:] - true_grad.data
#     if p == 1:
#         update = torch.sign(update)
#     elif p ==2:
#         update = update.view(n,-1)
#         update = update / (torch.norm(update, p=2, dim=1).view(n,1)+1e-6)
#     per_mask = per > eps
#     per_mask = per_mask.nonzero().view(-1)
#     # set overshoot for small pers
#     per[per_mask] = eps
#     X_adv = data.data + (((per+1e-4)*1.02).view(n,-1)*update.view(n,-1)).view(data.size())
#     #X_adv = data.data + (eps*update.view(n,-1)).view(data.size())
#     X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))
#     return X_adv

            
# def xp_attack_iter(model, data, target, eps, c=9 ,p=2, iter=100):
#     X_adv = data.cuda() + 0.0
#     update_num = 0.
#     for i in range(iter):
#         model.eval()
#         Xdata, Xtarget = X_adv, target.cuda()
#         Xdata, Xtarget = Variable(Xdata, requires_grad=True), Variable(Xtarget)
#         model.zero_grad()
#         Xoutput = model(Xdata)
#         Xpred = Xoutput.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         tmp_mask = Xpred.view_as(Xtarget)==Xtarget.data # get index
#         update_num += torch.sum(tmp_mask.long())
#         #print('need to attack: ', torch.sum(tmp_mask))
#         if torch.sum(tmp_mask.long()) < 1:
#             break
#         attack_mask = tmp_mask.nonzero().view(-1)
#         X_adv[attack_mask,:]  = xp_attack(model, X_adv[attack_mask,:], eps, c=c,p=p)
#         #print (i, ': ', torch.sum(tmp_mask.long()))
#     return X_adv.cpu(), update_num      











# #==============================================================================
# ## Our First Order Attack Adaptive
# #==============================================================================
# def xp_attack_adaptive(model, data, eps, c=9, p=2):
#     """Generate an adversarial pertubation using the xp method.
#     Pick the top false label and perturb towards that.
#     First-order attack

#     Args:
#         data: input image to perturb
#     """
    
#     """Generate an adversarial pertubation using the dp method.

#     Args:
#         data: input image to perturb
#     """
#     model.eval()
#     data = data.cuda()
#     data.requires_grad = True
#     model.zero_grad()
#     output = model(data)
    
#     output, ind = torch.sort(output, descending=True)
#     #c = output.size()[1]
#     n = len(data)

#     true_out = output[range(len(data)), n*[0]]
#     z_true = torch.sum(true_out)
#     data.grad = None
#     z_true.backward(retain_graph=True)
#     true_grad = data.grad
#     grads = torch.zeros([1+c] + list(data.size())).cuda()
#     pers = torch.zeros(len(data), 1+c).cuda()
#     for i in range(1,1+c):
#         z = torch.sum(output[:,i])
#         data.grad = None
#         model.zero_grad()
#         z.backward(retain_graph=True)
#         grad = data.grad+0. # batch_size x 3k
#         grads[i] = grad.data+0.
#         grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=p,dim=1) # batch_size x 1
#         pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
#     pers[range(n),n*[0]] = np.inf
#     pers[pers < 0] = 0
#     per, index = torch.min(pers,1) # batch_size x 1
#     #print('maximum pert: ', torch.max(per))
#     update = grads[index,range(len(data)),:] - true_grad.data
    
#     if p == 1:
#         update = torch.sign(update)
#     elif p ==2:
#         update = update.view(n,-1)
#         update = update / (torch.norm(update, p=2, dim=1).view(n,1)+1e-6)
    
#     #X_adv = data.data + (eps*update.view(n,-1)).view(data.size())
#     ### set large per to eps
#     per = per.view(n,-1)
#     per_mask = per > eps
#     per_mask = per_mask.nonzero().view(-1)
#     # set overshoot for small pers
#     per[per_mask,:] = eps[per_mask,:]
#     eps = per
#     X_adv = data.data + (1.02*(eps+1e-4)*update.view(n,-1)).view(data.size())
#     X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))


#     ### update eps magnitude
#     ori_diff = output[range(len(data)), n*[0]] - output[range(len(data)), n*[1]] 

#     adv_output = model(X_adv)    
#     adv_diff = adv_output[range(len(data)), ind[:,0]] - output[range(len(data)), ind[:,1]]

#     obj_diff = (ori_diff - adv_diff)/eps 

#     increase_ind = obj_diff > 1.0 
#     increase_ind = increase_ind.nonzero().view(-1)

#     decrease_ind = obj_diff < 0.1
#     decrease_ind = decrease_ind.nonzero().view(-1)

#     eps[increase_ind,:] = eps[increase_ind,:] * 1.2
#     eps[decrease_ind, :] = eps[decrease_ind,:] * 1.2

#     return X_adv, eps

            
# def xp_attack_adaptive_iter(model, data, target, eps, c=9 ,p=2, iter=100):
#     X_adv = data.cuda() + 0.0
#     update_num = 0.
#     eps = torch.from_numpy(np.array([eps]*len(data))).view(len(data), -1)
#     eps = eps.type(torch.FloatTensor).cuda()
#     for i in range(iter):
#         model.eval()
#         Xdata, Xtarget = X_adv, target.cuda()
#         Xdata, Xtarget = Variable(Xdata, requires_grad=True), Variable(Xtarget)
#         model.zero_grad()
#         Xoutput = model(Xdata)
#         Xpred = Xoutput.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         tmp_mask = Xpred.view_as(Xtarget)==Xtarget.data # get index
#         update_num += torch.sum(tmp_mask.long())
#         #print('need to attack: ', torch.sum(tmp_mask))
#         #print (i, ': ', torch.sum(tmp_mask.long()))
#         if torch.sum(tmp_mask.long()) < 1:
#             break
#         attack_mask = tmp_mask.nonzero().view(-1)
#         X_adv[attack_mask,:], eps[attack_mask,:]  = xp_attack_adaptive(model, X_adv[attack_mask,:], eps[attack_mask,:], c=c,p=p)
#     #print (i)
#     return X_adv.cpu(), update_num      












# #==============================================================================
# ## Our Second Order Attack with Lanczos Steps
# #==============================================================================
# def tridiag(a, b, c, k1=-1, k2=0, k3=1):
#     return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

# def processInput(c_a,c_b,c_c, eps):
#     bz = c_a.shape[0]
#     n = c_a.shape[1]
#     x = cvx.Variable(bz, n)
#     if np.max(eps) > 1e-6:
#         mag = 1./np.max(eps)     
#         obg = sum([x[i,:] *tridiag(c_b[i,:],c_a[i,:], c_b[i,:]) *x[i,:].T/2/mag/mag + c_c[i]*x[i,0]/mag for i in range(bz)])
#         cons = [cvx.sum_entries(cvx.square(x), axis=1) <= eps**2*mag*mag]
#         prob = cvx.Problem(cvx.Minimize(obg), cons)
#         qcqp = QCQP(prob)
#         qcqp.suggest(SDR)
#         f_cd, v_cd = qcqp.improve(COORD_DESCENT)

#         return x.value.A/mag
#     else:
#         return np.zeros([bz,n])

# def xp_attack_tr_lc_parallel(model, data, eps,c=9, lanczos_step=2):
#     """Generate an adversarial pertubation using the xp method.
#     TR-Lanczos

#     Args:
#         data: input image to perturb
#     """
#     model.eval()
#     data = data.cuda()
#     data.requires_grad = True
#     model.zero_grad()
#     output = model(data)
#     output = F.softmax(output)
    
#     output, ind = torch.sort(output, descending=True)
#     #c = output.size()[1]
#     n = len(data)
#     bz = n

#     true_out = output[range(len(data)), n*[0]]
#     z_true = torch.sum(true_out)
#     data.grad = None
#     z_true.backward(retain_graph=True)
#     true_grad = data.grad-0.
#     grads = torch.zeros([1+c] + list(data.size())).cuda()
#     pers = torch.zeros(len(data), 1+c).cuda()
#     for i in range(1,1+c):
#         z = torch.sum(output[:,i])
#         data.grad = None
#         model.zero_grad()
#         z.backward(retain_graph=True)
#         grad = data.grad+0. # batch_size x 3k
#         grads[i] = grad.data+0.
#         grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=2,dim=1) # batch_size x 1
#         pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
#     pers[range(n),n*[0]] = np.inf#1e16 some number is crazy
#     pers[pers < 0] = 0
#     per, index = torch.min(pers,1) # batch_size x 1
    
#     # initial perturbation
#     perturbation = torch.zeros(data.size()).cuda()
#     # print(pers) 
#     obj = torch.sum(output[range(bz), bz*[0]] - output[range(bz), index.long()])
#     obj.backward(create_graph=True)
#     gradients = data.grad+ 0. 
#     gradients = gradients.view(bz, -1)
#     grads_data = gradients.data + 0.
    
#     lanczos_linear = torch.norm(grads_data, p=2,dim=1).view(bz).cpu().numpy()
    
#     lanczos_vec = []
#     lanczos_diag = np.zeros([bz,lanczos_step])
#     lanczos_offdiag = np.zeros([bz,lanczos_step-1])
    
#     for i in range(lanczos_step):
#         u = data.grad.data.view(bz,-1) + 0.
#         # print('u norm: ', torch.norm(u))
#         if i == 0:
#             u = u / torch.norm(u, p=2, dim=1).view(bz,1)
#             lanczos_vec.append(u)
            
#         elif (i > 0) and (i < lanczos_step):
#             for j in range(i):
#                 u = u - lanczos_vec[j] * torch.sum(lanczos_vec[j] * u, dim=1).view(bz,1)
#                 u = u / torch.norm(u, p=2, dim=1).view(bz,1)
#                 lanczos_vec.append(u)


#         Hg = torch.sum(gradients * u)
#         data.grad = None
#         Hg.backward(retain_graph=True)
#         v = data.grad.data.view(bz,-1) + 0.       
#         if i == 0:
#             diag_tmp = torch.sum(u * v, dim=1).cpu().numpy()
#             lanczos_diag[:,i] = diag_tmp
#         else:
#             diag_tmp = torch.sum(u * v, dim=1).cpu().numpy()
#             lanczos_diag[:,i] = diag_tmp
#             offdiag_tmp = torch.sum(u*v_previous, dim=1).cpu().numpy()
#             lanczos_offdiag[:,i-1]=offdiag_tmp
            
#         v_previous = v + 0.
            

#     # replace not a number with inf
#     # print(lanczos_diag)

#     np.nan_to_num(np.inf)
#     lanczos_diag[np.isnan(lanczos_diag)] = np.inf
#     lanczos_offdiag[np.isnan(lanczos_offdiag)] = np.inf
#     lanczos_linear[np.isnan(lanczos_linear)] = np.inf

#     per_core = 1
#     parallel_num = int((bz+per_core-1)/per_core)
#     start_time = time.time()
#     sol = Parallel(n_jobs=parallel_num)(delayed(processInput)(lanczos_diag[i*per_core:(i+1)*per_core,:],lanczos_offdiag[i*per_core:(i+1)*per_core,:],lanczos_linear[i*per_core:(i+1)*per_core],np.array([eps for _ in range(per_core)])) for i in range(parallel_num))
#     #print(time.time()-start_time)
#     sol =  np.concatenate([t for t in sol], axis=0)
#     sol += 1e-4
#     vx = torch.zeros(data.size()).cuda()
#     vx = vx.view(bz,-1)
#     for i in range(lanczos_step):
#         sol_tmp = sol[:,i].reshape(bz,1)
#         sol_tmp = torch.from_numpy(sol_tmp)
#         sol_tmp = sol_tmp.type(torch.FloatTensor).cuda().contiguous()
#         vx = vx + sol_tmp * lanczos_vec[i]
    
#     vx = vx.view(data.size())

#     perturbation = vx
#     x_fgsm = data.data + 1.02*perturbation
#     X_adv = torch.clamp(x_fgsm, torch.min(data.data), torch.max(data.data))
#     return X_adv

# def xp_attack_tr_lc_parallel_iter(model, data, target, eps, c=9, iter=100, lanczos_step=2):
#     X_adv = data.cuda() + 0.0
#     update_num = 0.
#     for i in range(iter):
#         model.eval()
#         Xdata, Xtarget = X_adv, target.cuda()
#         Xdata, Xtarget = Variable(Xdata, requires_grad=True), Variable(Xtarget)
#         model.zero_grad()
#         Xoutput = model(Xdata)
#         Xpred = Xoutput.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         tmp_mask = Xpred.view_as(Xtarget)==Xtarget.data # get index
#         update_num += torch.sum(tmp_mask.long())
#         #print('need to attack: ', torch.sum(tmp_mask))
#         if torch.sum(tmp_mask.long()) < 1:
#             break
#         attack_mask = tmp_mask.nonzero().view(-1)
#         X_adv[attack_mask,:]= xp_attack_tr_lc_parallel(model, X_adv[attack_mask, :], eps, c=c, lanczos_step=lanczos_step)
#     return X_adv.cpu(), update_num 






# #==============================================================================
# ## Our First Order Attack with adaptive radius
# #==============================================================================

# def processInput_adaptive(c_a,c_b,c_c,c_d, eps):
#     bz = c_a.shape[0]
#     x = cvx.Variable(bz, 2)
#     if np.max(eps) > 1e-6:
#         tmp = eps + 0
#         tmp[eps<1e-6] = 1e16
#         mag = 1./np.min(tmp)     
#         obg = sum([x[i,:] *np.array([[c_b[i],c_d[i]], [c_d[i], c_c[i]]]) *x[i,:].T/2/mag/mag + c_a[i]*x[i,0]/mag for i in range(bz)])
#         cons = [cvx.sum_entries(cvx.square(x), axis=1) <= eps**2*mag*mag]
#         prob = cvx.Problem(cvx.Minimize(obg), cons)
#         qcqp = QCQP(prob)
#         qcqp.suggest(SDR)
#         f_cd, v_cd = qcqp.improve(COORD_DESCENT)

#         return x.value.A/mag
#     else:
#         return np.zeros([bz,2])

# def xp_attack_tr_lc_parallel_adaptive(model, data, target, eps_list, attack_target=None, strategy='random', islinear=True,threshould1=0.5, threshould2=1., eta=1.2):
#     """Generate an adversarial pertubation using the xp method.
#     TR-Lanczos

#     Args:
#         data: input image to perturb
#     """
#     model.eval()
#     # if not no_cuda:
#     data, target = data.cuda(), target.cuda()
#     data, target = Variable(data, requires_grad=True), Variable(target)
#     model.zero_grad()
#     output = model(data)
    
#     bz = data.size()[0]
    
# #     eps_list = torch.zeros(bz,1).cuda() + eps  # if already successed, need to set eps_list to zeros
#     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#     tmp_mask = pred.view_as(target)==target.data # get index
# #     print(type(eps_list), type(tmp_mask.float()))
#     eps_list = eps_list.cuda()
#     eps_list = eps_list * (tmp_mask.float().view_as(eps_list)) # if is not the same, set the perturbation to zero
#     eps_list = eps_list.cpu().numpy().reshape(bz,1)
    
#     if islinear:
#         output = F.log_softmax(output)

#     if strategy == 'random':
#         print('random attack')
#         Y_adv = generate_class(bz, target.data.cpu())
#         obj1 = output[range(bz), target.data] - output[range(bz), Y_adv.cuda().long()]
#         obj = torch.sum(output[range(bz), target.data] - output[range(bz), Y_adv.cuda().long()])
        
#     if strategy == 'target':
#         print('target attack')
#         obj1 = output[range(bz), target.data] - output[range(bz), attack_target]
#         obj = torch.sum(output[range(bz), target.data] - output[range(bz), attack_target])
        
#     elif strategy == 'second':
#         print('second largest attack')
#         sorted_out,ind = torch.sort(output,descending=True)
#         # encourage false labels
#         mask = ind[:,0].data == target.data
#         # minimize the objective
#         obj1 = sorted_out[range(len(data)),1-mask.long()] - sorted_out[range(len(data)),mask.long()]
#         obj = torch.sum(sorted_out[range(len(data)),1-mask.long()] - sorted_out[range(len(data)),mask.long()])
    
#     elif strategy == 'dp':
#         c = output.size()[1]
#         n = len(data)
#         true_out = output[range(len(data)), target.data]
#         z_true = torch.sum(true_out)
#         data.grad = None
#         z_true.backward(retain_graph=True)
#         true_grad = data.grad +0.
#         grads = torch.zeros([c] + list(data.size())).cuda()
#         pers = torch.zeros(len(data), c).cuda()
#         for i in range(c):
#             z = torch.sum(output[:,i])
#             data.grad = None
#             model.zero_grad()
#             z.backward(retain_graph=True)
#             grad = data.grad+0. # batch_size x 3k
#             grads[i] = grad.data +0.
#             grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=2,dim=1) # batch_size x 1
#             pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
#         pers[range(n),target.data] = 1e12
#         per, index = torch.min(pers,1) # batch_size x 1
# #         print(output[range(bz), index])
#         data.grad = None
#         obj1 = output[range(bz), target.data] - output[range(bz), index.long()]
#         obj = torch.sum(output[range(bz), target.data] - output[range(bz), index.long()])
#     else:
#         print('please choose a strategy')
#     obj.backward(create_graph=True)
#     gradients = data.grad +0.
#     gradients = gradients.view(bz, -1)
#     grads_data = gradients.data + 0.
#     u1 = gradients.data + 0
#     Hg = torch.sum(gradients * Variable(u1))
#     data.grad = None
#     Hg.backward(retain_graph=True)
#     Hessiang = data.grad.data.view(bz,-1) + 0
#     u2 = data.grad.data.view(bz,-1) + 0
#     u1 = u1 / torch.norm(u1, p=2, dim=1).view(bz,1)
#     u2 = u2 - u1 * torch.sum(u1 * u2, dim=1).view(bz,1)
#     u2 = u2 / torch.norm(u2, p=2, dim=1).view(bz,1)
    
#     # alpha * u1 + beta* u2
#     # a * alpha
#     coef_a = torch.sum(u1 * grads_data, dim=1).cpu().numpy()
#     # b * alpha * alpha
#     coef_b = torch.sum(u1 * Hessiang / torch.norm(grads_data, p=2, dim=1).view(bz,-1), dim=1).cpu().numpy()
#     # c * beta * beta
#     Hu2 = torch.sum(gradients * Variable(u2))
#     data.grad = None
#     Hu2.backward(retain_graph=True)
#     tmphu2 = data.grad.data.view(bz,-1)
#     coef_c = torch.sum(u2 * tmphu2, dim=1).cpu().numpy()
#     coef_d = torch.sum(u1 * tmphu2, dim=1).cpu().numpy()
#     per_core = 10
#     parallel_num = int(bz/per_core)
#     start_time = time.time()
#     sol = Parallel(n_jobs=parallel_num)(delayed(processInput_adaptive)(coef_a[per_core*i:per_core*(i+1)],coef_b[per_core*i:per_core*(i+1)],coef_c[per_core*i:per_core*(i+1)],coef_d[per_core*i:per_core*(i+1)], eps_list[per_core*i:per_core*(i+1),:]) for i in range(parallel_num))
#     print(time.time()-start_time)
#     sol = np.array(sol).reshape(bz,2)

#     sol1 = sol[:,0].reshape(bz,1)
#     sol2 = sol[:,1].reshape(bz,1)

#     sol1 = torch.from_numpy(sol1)
#     sol1 = sol1.type(torch.FloatTensor).cuda().contiguous()
    
#     sol2 = torch.from_numpy(sol2)
#     sol2 = sol2.type(torch.FloatTensor).cuda().contiguous()
#     vx = u1 * sol1 + u2 * sol2
    
#     vx = vx.view(data.size())
#     pertubation = vx
#     x_fgsm = data.data + pertubation
#     X_adv = torch.clamp(x_fgsm, torch.min(data.data), torch.max(data.data))
    
#     ### flatten grad_data_update  and X_adv - X 
#     grad_data_update = gradients + 0.
#     X_diff = (X_adv - data.data).view(bz,-1)
#     model_diff_grad = torch.sum(grad_data_update.data * X_diff, dim=1)
#     Hess_Xdiff = torch.sum(grad_data_update * Variable(X_diff))
#     data.grad = None
#     Hess_Xdiff.backward(retain_graph=True)    
#     Hx = data.grad.data.view(bz,-1)
#     xHx = torch.sum(Hx * X_diff, dim=1)
#     model_diff = xHx/2 + model_diff_grad
#     ### update eps_list and isallfooled
#     model.zero_grad()
#     output = model(Variable(X_adv))
#     # if already successed, need to set eps_list to zeros
#     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#     tmp_mask = pred.view_as(target)==target.data # get index
#     #     print(type(tmp_mask), type(tmp_mask.float()))
#     if torch.sum(tmp_mask.float()) < 1:
#         isallfooled = True
#     else:
#         isallfooled = False
        
#     if strategy == 'second':
#         sorted_out,ind = torch.sort(output,descending=True)
#         obj2 = sorted_out[range(len(data)),1-mask.long()] - sorted_out[range(len(data)),mask.long()] 
#     elif strategy == 'random':
#         obj2 = output[range(bz), Y_adv.cuda().long()] - output[range(bz), target.data]
#     elif strategy == 'dp':
#         obj2 = output[range(bz), index] - output[range(bz), target.data]
#     else:
#         print('wrong strategy')
#     obj_diff = obj2 - obj1
    
#     tr_ratio = obj_diff.data / (model_diff + 1e-6)
    
#     adaptive = torch.ones(bz,1).cuda()
#     # if tr_ratio < threshold1 eps / eta
#     adaptive[tr_ratio < threshould1] = 1/eta
#     # if tr_ratio > threshold2 eps * eta
#     adaptive[tr_ratio > threshould2] = eta
#     # otherwise
#     print('lower: ', torch.sum((tr_ratio < threshould1).float()), 'upper: ',torch.sum((tr_ratio > threshould2).float()))
    

#     max_eps = 0.05
#     min_eps = 0.0025
    
#     eps_list = eps_list * adaptive
#     eps_list[eps_list>max_eps] = max_eps
#     eps_list[eps_list<min_eps] = min_eps
#     return X_adv, eps_list, isallfooled
    
# def xp_attack_tr_lc_parallel_adaptive_iter(model, data, target, eps, islinear=True,attack_target=None, strategy='second', threshould1=0.5, threshould2=1., eta=1.2):
#     eps_list = torch.zeros(len(data),1).cuda() + eps
#     i = 0
#     while True:
#         data, eps_list, isallfooled = xp_attack_tr_lc_parallel_adaptive(model, data, target, eps_list, islinear=islinear, attack_target=attack_target, strategy=strategy, threshould1=threshould1, threshould2=threshould2, eta=eta)
#         if isallfooled:
#             break
#         print('iter: ', i+1)
#         i += 1
#     return data 
