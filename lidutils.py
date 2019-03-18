from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
#from utils import progress_bar
from progressbar import progress_bar

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
from models import *
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
# mnist roughly L2_difference/20
# cifar roughly L2_difference/54
# svhn roughly L2_difference/60
# be very carefully with these settings, tune to have noisy/adv have the same L2-norm
# otherwise artifact will lose its accuracy
# STDEVS = {
#     'mnist': {'fgsm': 0.264, 'bim-a': 0.111, 'bim-b': 0.184, 'cw-l2': 0.588},
#     'cifar': {'fgsm': 0.0504, 'bim-a': 0.0087, 'bim-b': 0.0439, 'cw-l2': 0.015},
#     'svhn': {'fgsm': 0.1332, 'bim-a': 0.015, 'bim-b': 0.1024, 'cw-l2': 0.0379}
# }

# fined tuned again when retrained all models with X in [-0.5, 0.5]
#STDEVS = {
#    'mnist': {'fgsm': 0.271, 'bim-a': 0.111, 'bim-b': 0.167, 'cw-l2': 0.207},
#    'cifar': {'fgsm': 0.104, 'fgm': 0.0504, 'bim-a': 0.0084, 'bim-b': 0.0428, 'cw-l2': 0.007},
#    'svhn': {'fgsm': 0.133, 'bim-a': 0.0155, 'bim-b': 0.095, 'cw-l2': 0.008}
#}

# CLIP_MIN = 0.0
# CLIP_MAX = 1.0
CLIP_MIN = -0.5
CLIP_MAX = 0.5
PATH_DATA = "data/"

# Set random seed
np.random.seed(0)

def transform_data(data, targets, batch_size):
    dataset = []
    for i in range(data.size()[0]):
        if (i+1)*batch_size < data.size()[0]:
            dataset.append((data[i*batch_size:(i+1)*batch_size], 
                            targets[i*batch_size:(i+1)*batch_size]))
        elif i*batch_size < data.size()[0]:
            dataset.append((data[i*batch_size:], 
                            targets[i*batch_size:]))
    return dataset

def lid_term(logits, batch_size=100):
    """Calculate LID loss term for a minibatch of logits

    :param logits: 
    :return: 
    """
    # y_pred = tf.nn.softmax(logits)
    y_pred = logits

    # calculate pairwise distance
    r = tf.reduce_sum(tf.square(y_pred), axis=1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(y_pred, tf.transpose(y_pred)) + tf.transpose(r)

    # find the k nearest neighbor
    D1 = tf.sqrt(D + 1e-9)
    D2, _ = tf.nn.top_k(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + 1e-9), axis=1)  # to avoid nan
    lids = -20 / v_log

    ## batch normalize lids
    # lids = tf.nn.l2_normalize(lids, dim=0, epsilon=1e-12)

    return lids

def lid_adv_term(clean_logits, adv_logits, batch_size=100):
    """Calculate LID loss term for a minibatch of advs logits

    :param logits: clean logits
    :param A_logits: adversarial logits
    :return: 
    """
    # y_pred = tf.nn.softmax(logits)
    c_pred = tf.reshape(clean_logits, (batch_size, -1))
    a_pred = tf.reshape(adv_logits, (batch_size, -1))

    # calculate pairwise distance
    r_a = tf.reduce_sum(tf.square(a_pred), axis=1)
    # turn r_a into column vector
    r_a = tf.reshape(r_a, [-1, 1])

    r_c = tf.reduce_sum(tf.square(c_pred), axis=1)
    # turn r_c into row vector
    r_c = tf.reshape(r_c, [1, -1])

    D = r_a - 2 * tf.matmul(a_pred, tf.transpose(c_pred)) + r_c

    # find the k nearest neighbor
    D1 = tf.sqrt(D + 1e-9)
    D2, _ = tf.nn.top_k(-D1, k=21, sorted=True)
    D3 = -D2[:, 1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + 1e-9), axis=1)  # to avoid nan
    lids = -20 / v_log

    ## batch normalize lids
    lids = tf.nn.l2_normalize(lids, dim=0, epsilon=1e-12)

    return lids

def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < CLIP_MAX)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = CLIP_MAX

    return np.reshape(x, original_shape)


def get_noisy_samples(X_test, X_test_adv, dataset, attack, eps):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw-l0']:
        X_test_noisy = torch.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
#             nb_diff = len(torch.where(X_test[i] != X_test_adv[i])[0])
            nb_diff = len(np.where(X_test[i].numpy() != X_test_adv[i].detach().numpy())[0])
            print(nb_diff)
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = torch.from_numpy(flip(X_test[i].detach().numpy(), nb_diff))
    else:
        warnings.warn("Important: using pre-set Gaussian scale sizes to craft noisy "
                      "samples. You will definitely need to manually tune the scale "
                      "according to the L2 print below, otherwise the result "
                      "will inaccurate. In future scale sizes will be inferred "
                      "automatically. For now, manually tune the scales around "
                      "mnist: L2/20.0, cifar: L2/54.0, svhn: L2/60.0")
        # Add Gaussian noise to the samples
        print('Adversarial Average L2 Norm: ',torch.mean(torch.norm((X_test_adv - X_test).data.view(X_test.size(0), -1), p=2, dim=1)))
        print(eps)
        X_test_noisy = torch.clamp(
            X_test + torch.tensor(np.random.normal(loc=0, scale=eps,
                                          size=X_test.size())).float(),
            torch.min(X_test.data), torch.max(X_test.data))
        print('Nosisy Average L2 Norm: ',torch.mean(torch.norm((X_test_noisy - X_test).data.view(X_test.size(0), -1), p=2, dim=1)))
    return X_test_noisy


def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    output_dim = model.layers[-1].output.shape[-1].value
    get_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output]
    )

    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = \
                get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)

def get_deep_representations(model, X, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -4
    output_dim = model.layers[-4].output.shape[-1].value
    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-4].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output

def get_layer_wise_activations(model, dataset):
    """
    Get the deep activation outputs.
    :param model:
    :param dataset: 'mnist', 'cifar', 'svhn', has different submanifolds architectures  
    :return: 
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # mnist model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    elif dataset == 'cifar':
        # cifar-10 model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    else:
        # svhn model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    return acts

# lid of a single query point x
def mle_single(data, x, k=20):
    #data = np.asarray(data, dtype=np.float32)
    #x = np.asarray(x, dtype=np.float32)
    data = data.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    #print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

# lid of a batch of query points X
def mle_batch(data, batch, k):
    #data = np.asarray(data, dtype=np.float32)
    #batch = np.asarray(batch, dtype=np.float32)
    data = data.cpu().detach().numpy()
    batch = batch.cpu().detach().numpy()
    
    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# mean distance of x to its k nearest neighbours
def kmean_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: np.mean(v)
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# mean distance of x to its k nearest neighbours
def kmean_pca_batch(data, batch, k=10):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    a = np.zeros(batch.shape[0])
    for i in np.arange(batch.shape[0]):
        tmp = np.concatenate((data, [batch[i]]))
        tmp_pca = PCA(n_components=2).fit_transform(tmp)
        a[i] = kmean_batch(tmp_pca[:-1], tmp_pca[-1], k=k)
    return a

def get_lids_random_batch(model, X, X_noisy, X_adv, targets, dataset, device, k=10, batch_size=100, detector_type='lid'):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model:
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images    
    :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures  
    :param k: the number of nearest neighbours for LID estimation  
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    # get deep representations
    model.module.change_mode('out_act')
    model.eval()
    test_normal = transform_data(X, targets, batch_size)
    test_noise = transform_data(X_noisy, targets, batch_size)
    test_adv = transform_data(X_adv, targets, batch_size)
    N = len(test_normal)
    with torch.no_grad():
        for batch_idx in range(N):
            (batch_normal, _) = test_normal[batch_idx]
            (batch_noise, _) = test_noise[batch_idx]
            (batch_adv, _) = test_adv[batch_idx]
            batch_normal = batch_normal.to(device)
            batch_noise = batch_noise.to(device)
            batch_adv = batch_adv.to(device)

            #Features using clean path
            out_normal, act_normal = model(batch_normal)
            out_noise, act_noise = model(batch_noise)
            out_adv, act_adv = model(batch_adv)
            
            if detector_type == 'lid': 
                n = act_normal[0].size()[0]
                lid_batch = torch.zeros(n, len(act_normal))
                lid_batch_adv = torch.zeros(n, len(act_normal))
                lid_batch_noisy = torch.zeros(n, len(act_normal))
                for i in range(len(act_normal)):
                    lid_batch[:, i] = torch.FloatTensor(mle_batch(act_normal[i], act_normal[i], k=k))
                    lid_batch_adv[:, i] = torch.FloatTensor(mle_batch(act_normal[i], act_adv[i], k=k))
                    lid_batch_noisy[:, i] = torch.FloatTensor(mle_batch(act_normal[i], act_noise[i], k=k))
            
            else:
                raise('do not know type')
            #Concat the features
            if batch_idx == 0:
                lids = lid_batch
                lids_adv = lid_batch_adv
                lids_noisy = lid_batch_noisy
            else:
                lids = torch.cat((lids, lid_batch))
                lids_adv = torch.cat((lids_adv, lid_batch_adv))
                lids_noisy = torch.cat((lids_noisy, lid_batch_noisy))
            progress_bar(batch_idx, N)
        return lids, lids_noisy, lids_adv
        
def get_kmeans_random_batch(model, X, X_noisy, X_adv, dataset, k=10, batch_size=100, pca=False):
    """
    Get the mean distance of each Xi in X_adv to its k nearest neighbors.

    :param model:
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images    
    :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures  
    :param k: the number of nearest neighbours for LID estimation  
    :param batch_size: default 100
    :param pca: using pca or not, if True, apply pca to the referenced sample and a 
            minibatch of normal samples, then compute the knn mean distance of the referenced sample.
    :return: kms_normal: kmean of normal images (num_examples, 1)
            kms_noisy: kmean of normal images (num_examples, 1)
            kms_adv: kmean of adv images (num_examples, 1)
    """
    # get deep representations
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])]
    km_dim = len(funcs)
    print("Number of layers to use: ", km_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        km_batch = np.zeros(shape=(n_feed, km_dim))
        km_batch_adv = np.zeros(shape=(n_feed, km_dim))
        km_batch_noisy = np.zeros(shape=(n_feed, km_dim))
        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_act: ", X_act.shape)

            X_adv_act = func([X_adv[start:end], 0])[0]
            X_adv_act = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_adv_act: ", X_adv_act.shape)

            X_noisy_act = func([X_noisy[start:end], 0])[0]
            X_noisy_act = np.asarray(X_noisy_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_noisy_act: ", X_noisy_act.shape)

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            if pca:
                km_batch[:, i] = kmean_pca_batch(X_act, X_act, k=k)
            else:
                km_batch[:, i] = kmean_batch(X_act, X_act, k=k)
            # print("lid_batch: ", lid_batch.shape)
            if pca:
                km_batch_adv[:, i] = kmean_pca_batch(X_act, X_adv_act, k=k)
            else:
                km_batch_adv[:, i] = kmean_batch(X_act, X_adv_act, k=k)
            # print("lid_batch_adv: ", lid_batch_adv.shape)
            if pca:
                km_batch_noisy[:, i] = kmean_pca_batch(X_act, X_noisy_act, k=k)
            else:
                km_batch_noisy[:, i] = kmean_batch(X_act, X_noisy_act, k=k)
                # print("lid_batch_noisy: ", lid_batch_noisy.shape)
        return km_batch, km_batch_noisy, km_batch_adv

    kms = []
    kms_adv = []
    kms_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        km_batch, km_batch_noisy, km_batch_adv = estimate(i_batch)
        kms.extend(km_batch)
        kms_adv.extend(km_batch_adv)
        kms_noisy.extend(km_batch_noisy)
        # print("kms: ", kms.shape)
        # print("kms_adv: ", kms_noisy.shape)
        # print("kms_noisy: ", kms_noisy.shape)

    kms = np.asarray(kms, dtype=np.float32)
    kms_noisy = np.asarray(kms_noisy, dtype=np.float32)
    kms_adv = np.asarray(kms_adv, dtype=np.float32)

    return kms, kms_noisy, kms_adv

def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """Z-score normalisation
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(X, y, class_type='lr'):
    """
    TODO
    :param X: the data samples
    :param y: the labels
    :return:
    """
    #LID classifier
    if class_type == 'lr':
        lr = LogisticRegressionCV(n_jobs=-1).fit(X, y) 
    #lr = SVC(gamma=1).fit(X, y)
    #lr = RandomForestClassifier().fit(X, y)
    else:
        lr = MLPClassifier(hidden_layer_sizes=(64, 32, 16,), activation='tanh').fit(X, y)
    return lr


def train_lr_rfeinman(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr


def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def compute_roc_rfeinman(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def random_split(X, Y):
    """
    Random split the data into 80% for training and 20% for testing
    :param X: 
    :param Y: 
    :return: 
    """
    print("random split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    num_train = int(num_samples * 0.8)
    rand_pert = np.random.permutation(num_samples)
    X = X[rand_pert]
    Y = Y[rand_pert]
    X_train, X_test = X[:num_train], X[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    return X_train, Y_train, X_test, Y_test

def block_split(X, Y):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 100.
    :param X: 
    :param Y: 
    :return: 
    """
    print("Isolated split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.008) * 100

    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    # unit test
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    c = np.array([11, 12, 13, 14, 15])

    a_z, b_z, c_z = normalize(a, b, c)
    print(a_z)
    print(b_z)
    print(c_z)
