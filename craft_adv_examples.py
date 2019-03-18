from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np

from models import *
from attack_method_lid import fgsm, fgsm_adaptive_iter, deep_fool_iter 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import foolbox as fb

from utils import *
from lidutils import *
def craft_one_type(model, testloader, dataset, attack, batch_size, device):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """
    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 0:
                Adv_data = fgsm(
                model, inputs, targets, eps=args.attack_eps)
                Adv_targets = targets
                X = inputs
            else:
                Adv_data = torch.cat((Adv_data, fgsm(
                    model, inputs, targets, eps=args.attack_eps)), dim=0)
                Adv_targets = torch.cat((Adv_targets, targets), dim=0)
                X = torch.cat((X, inputs), dim=0)
         #TODO: Implement iteration version   
    elif attack == 'ifgsm':
        # FGSM_iter attack
        print('Crafting fgsm_iter adversarial samples...')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 0:
                Adv_data = fgsm_adaptive_iter(
                model, inputs, targets, eps=args.attack_eps, iter=200)
                Adv_targets = targets
                X = inputs
            else:
                Adv_data = torch.cat((Adv_data, fgsm_adaptive_iter(
                    model, inputs, targets, eps=args.attack_eps, iter=200)), dim=0)
                Adv_targets = torch.cat((Adv_targets, targets), dim=0)
                X = torch.cat((X, inputs), dim=0)
    #TODO: Implement deep fool and tr_attack
    elif attack == 'deep_fool_l2':
        # DeepFool attack
        # here c means how many candidate we want to consider, p means the attack norm, worst_case is useless, set it to False
        print('Crafting DeepFool-l2 adversarial samples...')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 0:
                Adv_data = deep_fool_iter(
                    model, inputs, targets, c=9, p=2, iter=1000, worst_case = False)
                Adv_targets = targets
                X = inputs
            else:
                Adv_data = torch.cat((Adv_data, deep_fool_iter(
                    model, inputs, targets, c=9, p=2, iter=1000, worst_case = False)), dim=0)
                Adv_targets = torch.cat((Adv_targets, targets), dim=0)
                X = torch.cat((X, inputs), dim=0)
    elif attack == 'deep_fool_inf':
        # DeepFool attack
        # here c means how many candidate we want to consider, p means the attack norm, worst_case is useless, set it to False
        print('Crafting DeepFool-l2 adversarial samples...')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if batch_idx == 0:
                Adv_data = deep_fool_iter(
                    model, inputs, targets, c=9, p=1, iter=1000, worst_case = False)
                Adv_targets = targets
                X = inputs
            else:
                Adv_data = torch.cat((Adv_data, deep_fool_iter(
                    model, inputs, targets, c=9, p=1, iter=1000, worst_case = False)), dim=0)
                Adv_targets = torch.cat((Adv_targets, targets), dim=0)
                X = torch.cat((X, inputs), dim=0)
    elif attack == 'cw-l2':
        # CW attack
        # here c means how many candidate we want to consider, p means the attack norm, worst_case is useless, set it to False
        print('Crafting CW Attack adversarial samples...')
        fb_model = fb.models.PyTorchModel(model, bounds=(CLIP_MIN,CLIP_MAX), num_classes=10)
        cw_attack = fb.attacks.CarliniWagnerL2Attack(fb_model)
        if args.batch_size != 1:
            raise('If you want to use CW attack, the batch size must be 1')
        if args.dataset == 'cifar10':
            Adv_data = torch.Tensor(len(testloader),3,32,32).to(device)
            X = torch.Tensor(len(testloader),3,32,32).to(device)
        else:
            Adv_data = torch.Tensor(len(testloader),1,28,28).to(device)
            X = torch.Tensor(len(testloader),1,28,28).to(device)
        Adv_targets = torch.LongTensor(len(testloader)).to(device)
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            print(batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            Adv_data[batch_idx,] = torch.from_numpy(
                cw_attack(inputs.cpu().numpy()[0,:], targets.cpu().numpy()[0], learning_rate=0.01)).to(device)
            Adv_targets[batch_idx] = targets[0]
            X[batch_idx] = inputs[0,:]
    else:
        raise("we did not support this attack yet")
    #TODO: temporary used for testing
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    testset = transform_data(Adv_data, Adv_targets, args.batch_size)
    #testset = transform_data(X, Adv_targets, args.batch_size)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testset):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
        print("Model accuracy on the adversarial test set: %0.2f%%" % acc)

    #Bring X and X_adv to cpu
    Adv_data = Adv_data.cpu()
    Adv_targets = Adv_targets.cpu()
    X = X.cpu()
    #Saving adversarial example
    adv_file = os.path.join(args.data_path, "Adv_%s_%s.pth" % (dataset, attack))
    torch.save([Adv_data, Adv_targets], adv_file)
    #l2_diff = F.mse_loss(Adv_data.view(X.size(0), -1), X.view(X.size(0), -1))
    l2_diff = 0.
    for i in range(X.size(0)):
        l2_diff += torch.norm(Adv_data[i, :] - X[i, :], p=2)
    l2_diff /= X.size(0)
    print("Average L-2 perturbation size of the %s attack: %0.2f" %
          (attack, l2_diff))

def main(args):
    assert args.dataset in ['mnist', 'cifar10', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'ifgsm', 'deep_fool_l2', 'deep_fool_inf', 'cw-l2'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2', 'all' or 'cw-lid' for attacking LID detector"
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.cuda = True if torch.cuda.is_available() else False
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    
    #================================================
    # get model
    #================================================
    model_list = {
            'LeNetLike': LeNetLike(jump = args.jump),
            'AlexLike': AlexLike(jump = args.jump),
            #'JumpResNet': JumpResNet(depth=20, jump = args.jump),
            'MobileNetV2': MobileNetV2(jump = args.jump),      
    }
    
    
    model = model_list[args.arch]
    if args.cuda:
        model.cuda()
    
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    
    #Initial test
    trainloader, testloader = getData(name=args.dataset, train_bs=args.batch_size, test_bs=args.batch_size)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #_, acc = test(0, model, optimizer, criterion, testloader, device)
    #print("Accuracy on the test set: %0.2f%%" % acc)
    #TODO: temporary used for testing
    #criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
        print("Model accuracy on the adversarial test set: %0.2f%%" % acc)
    '''
    if args.attack == 'cw-lid': # white box attacking LID detector - an example
        X_test = X_test[:1000]
        Y_test = Y_test[:1000]
    '''
    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2']:
            craft_one_type(model, testloader, args.dataset, attack,
                           args.batch_size, device)
    else:
        # Craft one specific attack type
        craft_one_type(model, testloader, args.dataset, args.attack,
                       args.batch_size, device)
    print('Adversarial samples crafted and saved to %s ' % args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar10' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'deep_fool', 'tr_attack', 'jsma' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '--arch',
        help="arch",
        required=True, type=str
    )
    parser.add_argument(
        '--resume',
        help="The pre-trained model",
        required=True, type=str
    )
    parser.add_argument('-ae', '--attack-eps', 
            default=0.01, 
            required=False, type=float)
    parser.add_argument(
        '-dp', '--data_path',
        help="where do you want to store data",
        required=True, type=str
    )
    parser.add_argument('--jump', 
            #default=0.01, 
            required=True, type=float)
    parser.set_defaults(batch_size=100)
    args = parser.parse_args()
    main(args)
