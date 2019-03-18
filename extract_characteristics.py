from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from copy import deepcopy

from models import *
from attack_method_lid import fgsm, fgsm_adaptive_iter, deep_fool_iter 
from utils import *
from progressbar import progress_bar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from lidutils import (get_noisy_samples, get_mc_predictions,
                      get_deep_representations, score_samples, normalize,
                      get_lids_random_batch, get_kmeans_random_batch, transform_data)

# In the original paper, the author used optimal KDE bandwidths dataset-wise
#  that were determined from CV tuning
BANDWIDTHS = {'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00}

#TODO: Only modify the LID feature extraction

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    #X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.size())
    X_pos = X_pos.view((X_pos.size()[0], -1))

    #X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.size())
    X_neg = X_neg.view((X_neg.size()[0], -1))

    X = torch.cat((X_pos, X_neg))
    y = torch.cat((torch.ones(X_pos.size()[0]), torch.zeros(X_neg.size()[0])))
    y = y.view((X.size()[0], 1))

    return X, y


def get_kd(model, X_train, Y_train, X_test, X_test_noisy, X_test_adv):
    """
    Get kernel density scores
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with kd values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train,
                                                batch_size=args.batch_size)
    X_test_normal_features = get_deep_representations(model, X_test,
                                                      batch_size=args.batch_size)
    X_test_noisy_features = get_deep_representations(model, X_test_noisy,
                                                     batch_size=args.batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                   batch_size=args.batch_size)
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    print('bandwidth %.4f for %s' % (BANDWIDTHS[args.dataset], args.dataset))
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[args.dataset]) \
            .fit(X_train_features[class_inds[i]])
    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = model.predict_classes(X_test, verbose=0,
                                              batch_size=args.batch_size)
    preds_test_noisy = model.predict_classes(X_test_noisy, verbose=0,
                                             batch_size=args.batch_size)
    preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
                                           batch_size=args.batch_size)
    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_noisy = score_samples(
        kdes,
        X_test_noisy_features,
        preds_test_noisy
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )

    print("densities_normal:", densities_normal.shape)
    print("densities_adv:", densities_adv.shape)
    print("densities_noisy:", densities_noisy.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
    #     densities_normal,
    #     densities_adv,
    #     densities_noisy
    # )

    densities_pos = densities_adv
    densities_neg = np.concatenate((densities_normal, densities_noisy))
    artifacts, labels = merge_and_generate_labels(densities_pos, densities_neg)

    return artifacts, labels

def get_bu(model, X_test, X_test_noisy, X_test_adv):
    """
    Get Bayesian uncertainty scores
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with bu values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(model, X_test,
                                        batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_noisy = get_mc_predictions(model, X_test_noisy,
                                       batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(model, X_test_adv,
                                     batch_size=args.batch_size) \
        .var(axis=0).mean(axis=1)

    print("uncerts_normal:", uncerts_normal.shape)
    print("uncerts_noisy:", uncerts_noisy.shape)
    print("uncerts_adv:", uncerts_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
    #     uncerts_normal,
    #     uncerts_adv,
    #     uncerts_noisy
    # )

    uncerts_pos = uncerts_adv
    uncerts_neg = np.concatenate((uncerts_normal, uncerts_noisy))
    artifacts, labels = merge_and_generate_labels(uncerts_pos, uncerts_neg)

    return artifacts, labels

def get_lid(model, X_test, X_test_noisy, X_test_adv, X_targets, device, k=10, batch_size=100, dataset='mnist', detector_type='lid'):
    """
    Get local intrinsic dimensionality
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(model, X_test, X_test_noisy,
                                                              X_test_adv, X_targets, dataset, device, 
                                                              k, batch_size, detector_type)
    print("lids_normal:", lids_normal.size())
    print("lids_noisy:", lids_noisy.size())
    print("lids_adv:", lids_adv.size())

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # lids_normal_z, lids_adv_z, lids_noisy_z = normalize(
    #     lids_normal,
    #     lids_adv,
    #     lids_noisy
    # )

    lids_pos = lids_adv
    lids_neg = torch.cat((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

def get_kmeans(model, X_test, X_test_noisy, X_test_adv, k=10, batch_size=100, dataset='mnist'):
    """
    Calculate the average distance to k nearest neighbours as a feature.
    This is used to compare density vs LID. Why density doesn't work?
    :param model: 
    :param X_train: 
    :param Y_train: 
    :param X_test: 
    :param X_test_noisy: 
    :param X_test_adv: 
    :return: artifacts: positive and negative examples with lid values, 
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract k means feature: k = %s' % k)
    kms_normal, kms_noisy, kms_adv = get_kmeans_random_batch(model, X_test, X_test_noisy,
                                                              X_test_adv, dataset, k, batch_size,
                                                             pca=True)
    print("kms_normal:", kms_normal.shape)
    print("kms_noisy:", kms_noisy.shape)
    print("kms_adv:", kms_adv.shape)

    ## skip the normalization, you may want to try different normalizations later
    ## so at this step, just save the raw values
    # kms_normal_z, kms_noisy_z, kms_adv_z = normalize(
    #     kms_normal,
    #     kms_noisy,
    #     kms_adv
    # )

    kms_pos = kms_adv
    kms_neg = np.concatenate((kms_normal, kms_noisy))
    artifacts, labels = merge_and_generate_labels(kms_pos, kms_neg)

    return artifacts, labels

def main(args):
    assert args.dataset in ['mnist', 'cifar10', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'ifgsm', 'deep_fool_l2', 'deep_fool_inf', 'cw-l2'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2', 'all' or 'cw-lid' for attacking LID detector"
    assert args.characteristic in ['kd', 'bu', 'lid', 'km', 'all'], \
        "Characteristic(s) to use 'kd', 'bu', 'lid', 'km', 'all'"
    adv_file = os.path.join(args.data_path, "Adv_%s_%s.pth" % (args.dataset, args.attack))
    assert os.path.isfile(adv_file), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_samples.py'

    print('Loading the data and model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.cuda = True if torch.cuda.is_available() else False
    print(args.basic_lid) 
    #Load model file, model is in checkpoint['net']

    #================================================
    # get model
    #================================================
    model_list = {
            'LeNetLike': LeNetLike(jump = args.jump),
            'AlexLike': AlexLike(jump = args.jump),
            #'JumpResNet': JumpResNet(depth=20, jump = args.jump),
            'MobileNetV2': MobileNetV2(jump = args.jump),      
            'WideResNet': WideResNet(depth=args.depth, widen_factor=args.widen_factor, dropout_rate=args.dropout, num_classes=10, level=1, jump=args.jump), 
    }
    
    
    model = model_list[args.arch]
    if args.cuda:
        model.cuda()
    
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    
    #Initial test
    trainloader, testloader = getData(name=args.dataset, train_bs=1, test_bs=1)
    
    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    if args.attack == 'all':
        # TODO: implement 'all' option
        # X_test_adv = ...
        # X_test_noisy = ...
        raise NotImplementedError("'All' types detector not yet implemented.")
    else:
        # Load adversarial samples
        X_adv = torch.load(adv_file)
        X_test_adv = X_adv[0]
        X_targets = X_adv[1]
        print("X_test_adv: ", X_test_adv.size())

        # as there are some parameters to tune for noisy example, so put the generation
        # step here instead of the adversarial step which can take many hours
        noisy_file = os.path.join(args.data_path, 'Noisy_%s_%s.pth' % (args.dataset, args.attack))
        X_test = torch.empty(X_test_adv.size())
        for i, (x, _) in enumerate(testloader):
            X_test[i] = x.clone()
        #if os.path.isfile(noisy_file):
        if False:
            X_test_noisy = torch.load(noisy_file)
        else:
            # Craft an equal number of noisy samples
            print('Crafting %s noisy samples. ' % args.dataset)
            
            
            X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset, args.attack, args.noisy_eps)
            torch.save(X_test_noisy, noisy_file)

    # Check model accuracies on each sample type

    for s_type in ['normal', 'noisy', 'adversarial']:     
        criterion = nn.CrossEntropyLoss()
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        if s_type == 'adversarial':
            testset = transform_data(X_test_adv, X_targets, args.batch_size)
        elif s_type == 'noisy':
            testset = transform_data(X_test_noisy, X_targets, args.batch_size)
        else:
            trainloader, testset = getData(name=args.dataset, train_bs=100, test_bs=100)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testset):
                inputs, targets = inputs.to(device), targets.to(device)
                # test acc after recons
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if s_type == 'normal':
                    if batch_idx == 0:
                        inds_correct = batch_idx*args.batch_size+((predicted-targets) == 0).nonzero().squeeze()
                    else:
                        inds_correct = torch.cat((inds_correct, batch_idx*args.batch_size+
                                              ((predicted-targets) == 0).nonzero().squeeze()))

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            acc = 100.*correct/total
            print("Model accuracy on the %s test set: %0.2f%%" % (s_type, acc))
    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    print("Number of correctly predict images: %s" % (len(inds_correct)))

    X_test = X_test[inds_correct]
    X_test_noisy = X_test_noisy[inds_correct]
    X_test_adv = X_test_adv[inds_correct]
    Y_correct = X_targets[inds_correct]
    torch.save(X_test_adv, args.data_path+'/Adv_%s_%s_correct.pth' % (args.dataset, args.attack))
    torch.save(X_test, args.data_path+'/Normal_%s_%s_correct.pth' % (args.dataset, args.attack))
    torch.save(X_test_noisy, args.data_path+'/Noisy_%s_%s_correct.pth' % (args.dataset, args.attack))
    torch.save(Y_correct, args.data_path+'/Label_correct.pth')
    print("X_test: ", X_test.shape)
    print("X_test_noisy: ", X_test_noisy.shape)
    print("X_test_adv: ", X_test_adv.shape)

    #TODO: Implement this
    if args.characteristic == 'kd':
        # extract kernel density
        characteristics, labels = get_kd(correct_model, X_train, Y_train, X_test, X_test_noisy, X_test_adv)
        print("KD: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        bandwidth = BANDWIDTHS[args.dataset]
        file_name = os.path.join(args.data_path, 'kd_%s_%s_%.4f.npy' % (args.dataset, args.attack, bandwidth))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'bu':
        # extract Bayesian uncertainty
        characteristics, labels = get_bu(correct_model, X_test, X_test_noisy, X_test_adv)
        print("BU: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        file_name = os.path.join(args.data_path, 'bu_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'lid':
        # extract local intrinsic dimensionality
        characteristics, labels = get_lid(model, X_test, X_test_noisy, X_test_adv, X_targets, device,  
                                    args.k_nearest, args.batch_size, args.dataset, args.basic_lid)
        #TODO: neet to change
        print("LID: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        # file_name = os.path.join(args.data_path, 'lid_%s_%s.npy' % (args.dataset, args.attack))
        #file_name = os.path.join('./data_grid_search/lid_large_batch/', 'lid_%s_%s_%s.pth' %
        #                         (args.dataset, args.attack, args.k_nearest))
        file_name = os.path.join(args.data_path, 'lid_%s_%s_%s.pth_%s' %
                                 (args.dataset, args.attack, args.k_nearest, args.basic_lid))

        data = torch.cat((characteristics, labels), dim=1)
        torch.save(data, file_name)
    elif args.characteristic == 'km':
        # extract k means distance
        characteristics, labels = get_kmeans(correct_model, X_test, X_test_noisy, X_test_adv,
                                    args.k_nearest, args.batch_size, args.dataset)
        print("K-Mean: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)

        # save to file
        file_name = os.path.join(args.data_path, 'km_pca_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    elif args.characteristic == 'all':
        # extract kernel density
        characteristics, labels = get_kd(correct_model, X_train, Y_train, X_test, X_test_noisy, X_test_adv)
        file_name = os.path.join(args.data_path, 'kd_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)

        # extract Bayesian uncertainty
        characteristics, labels = get_bu(correct_model, X_test, X_test_noisy, X_test_adv)
        file_name = os.path.join(args.data_path, 'bu_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)

        # extract local intrinsic dimensionality
        characteristics, labels = get_lid(correct_model, X_test, X_test_noisy, X_test_adv,
                                    args.k_nearest, args.batch_size, args.dataset)
        file_name = os.path.join(args.data_path, 'lid_%s_%s.npy' % (args.dataset, args.attack))
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)

        # extract k means distance
        # artifcharacteristics, labels = get_kmeans(model, X_test, X_test_noisy, X_test_adv,
        #                                args.k_nearest, args.batch_size, args.dataset)
        # file_name = os.path.join(args.data_path, 'km_%s_%s.npy' % (args.dataset, args.attack))
        # data = np.concatenate((characteristics, labels), axis=1)
        # np.save(file_name, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'select_index', 'deep_fool', 'tr_attack' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--characteristic',
        help="Characteristic(s) to use 'kd', 'bu', 'lid' 'km' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-k', '--k_nearest',
        help="The number of nearest neighbours to use; either 10, 20, 100 ",
        required=False, type=int
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
    parser.add_argument(
        '-bl', '--basic_lid',
        help="using classifier or coder",
        default='lid',
        required=False, type=str
    )
    parser.add_argument('-ne', '--noisy-eps', 
            default=0.01, 
            required=True, type=float)
    parser.add_argument(
        '-dp', '--data_path',
        help="where do you want to store data",
        required=True, type=str
    )
    parser.add_argument('--jump', 
            #default=0.01, 
            required=True, type=float)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(k_nearest=20)
    args = parser.parse_args()
    main(args)
