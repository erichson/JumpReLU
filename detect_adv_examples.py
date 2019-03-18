from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from lidutils import (random_split, block_split, train_lr, compute_roc)
import torch
from joblib import dump, load
from models import *
from lidutils import *

DATASETS = ['mnist', 'cifar10', 'svhn']
ATTACKS = ['fgsm', 'ifgsm', 'deep_fool_l2', 'deep_fool_inf', 'cw-l2']
CHARACTERISTICS = ['lid']
#args.data_path = "data/"
PATH_IMAGES = "plots/"

def load_characteristics(dataset, attack, characteristics):
    """
    Load multiple characteristics for one dataset and one attack.
    :param dataset: 
    :param attack: 
    :param characteristics: 
    :return: 
    """
    X, Y = None, None
    for characteristic in characteristics:
        # print("  -- %s" % characteristics)
        file_name = os.path.join(args.data_path_characteristics)
        data = torch.load(file_name)
        if X is None:
            X = data[:, :-1]
        else:
            X = np.concatenate((X, data[:, :-1]), axis=1)
        if Y is None:
            Y = data[:, -1] # labels only need to load once

    return X, Y

def generate_classify(args, X, Y, device='cuda'):
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    N = Y.size()[0]
    testloader = transform_data(X, Y, 100)
    Label = torch.zeros(Y.size()[0]).to(device)
    
    checkpoint = torch.load('checkpoint/resnet18.pth_0')
    model = ResNet18()
    ae = ResNet18()
    model.load_state_dict(checkpoint['net'])

    # using autoencoder
    checkpoint = torch.load('checkpoint/resnet18.pth_0')
    ae.load_state_dict(checkpoint['net'])
    #Parallel the model
    model = model.to(device)
    ae = ae.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        ae = torch.nn.DataParallel(ae)
        cudnn.benchmark = True

    ae.eval()
    model.eval()
    
    print('Generate Labels for Adv')  
    for batch_idx, (inputs, targets) in enumerate(testloader):
        n = targets.size()[0]
        label = torch.zeros(n).to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predict1 = outputs.max(1)
        outputs = ae(inputs)
        _, predict2 = outputs.max(1)
        idx = ((predict2-targets) == 0).nonzero()
        label[idx] = 2
        idx = ((predict1-targets) == 0).nonzero()
        label[idx] = 1
        Label[batch_idx*100:batch_idx*100+n] = label.clone()
 
    print('Saving Labels >>>>') 
    Label = Label.cpu()
    return Label
        

def detect(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ATTACKS, \
        "Train attack must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2'"
    assert args.test_attack in ATTACKS, \
        "Test attack must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma', 'cw-l2'"
    characteristics = args.characteristics.split(',')
    for char in characteristics:
        assert char in CHARACTERISTICS, \
            "Characteristic(s) to use 'kd', 'bu', 'lid'"

    print("Loading train attack: %s" % args.attack)
    X, Y = load_characteristics(args.dataset, args.attack, characteristics)

    # standarization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    # X = scale(X) # Z-norm

    # test attack is the same as training attack
    X_train, Y_train, X_test, Y_test = block_split(X, Y)
    
    if args.test_attack != args.attack:
        # test attack is a different attack
        print("Loading test attack: %s" % args.test_attack)
        X_test, Y_test = load_characteristics(args.dataset, args.test_attack, characteristics)
        _, _, X_test, Y_test = block_split(X_test, Y_test)

        # apply training normalizer
        # X_test = scaler.transform(X_test)
        # X_test = scale(X_test) # Z-norm

    print("Train data size: ", X_train.shape)
    print("Test data size: ", X_test.shape)


    ## Build detector
    print("%s Detector on [dataset: %s, train_attack: %s, test_attack: %s] with:" %
                                        (args.classifier_type, args.dataset, args.attack, args.test_attack))
    
    #Save model
    lr = train_lr(X_train, Y_train, args.classifier_type)
    dump(lr, args.data_path+'/classifier.joblib'+args.classifier_type)
    

    ## Evaluate detector
    y_pred = lr.predict_proba(X_test)[:, 1]
    y_label_pred = lr.predict(X_test)
    
    # AUC
    _, _, auc_score = compute_roc(Y_test, y_pred, plot=False)
    precision = precision_score(Y_test, y_label_pred)
    recall = recall_score(Y_test, y_label_pred)

    y_label_pred = lr.predict(X_test)
    acc = accuracy_score(Y_test, y_label_pred)
    print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, acc, precision, recall))
    ### used for recored result into a txt file
    f = open('results.txt', 'a+')
    #f.write(args.classifier_type+'\n')
    #f.write(str(X_train.shape)+'\n')
    f.write('&'+str(round(auc_score*10000)/100)+'/'+str(round(acc*10000)/100)+'\% ')
    f.close()    
    ### end writing

    return lr, auc_score, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use train the discriminator; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw-l2'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--characteristics',
        help="Characteristic(s) to use any combination in ['kd', 'bu', 'lid'] "
             "separated by comma, for example: kd,bu",
        required=True, type=str
    )
    parser.add_argument(
        '-t', '--test_attack',
        help="Characteristic(s) to cross-test the discriminator.",
        required=False, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-ct', '--classifier-type',
        help="classifier type: lr mlp",
        required=True, type=str
    )
    parser.add_argument(
        '-dp', '--data_path',
        help="where do you want to store data",
        required=True, type=str
    )
    parser.add_argument(
        '-dpc', '--data_path_characteristics',
        help="where do you want to load characteristic",
        required=True, type=str
    )
    
    parser.set_defaults(batch_size=100)
    parser.set_defaults(test_attack=None)
    args = parser.parse_args()
    detect(args)
