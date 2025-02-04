import os
from tqdm import tqdm

# import cv2
import numpy as np
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# from cluster import ElasticNetSubspaceClustering, clustering_accuracy
import utils

    
def get_features(net, trainloader, loss_name, model_name, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, data in enumerate(train_bar):
        if model_name == 'mvcnn':
            N, V, C, H, W = data[1].size()
            batch_imgs = data[1].view(-1, C, H, W)
        else:
            batch_imgs = data[1]
        batch_lbls = data[0].long()

        if loss_name == 'mcr2':
            batch_features = net(batch_imgs.cpu())
        else:  # 'ce'
            _, batch_features = net(batch_imgs.cpu())

        features.append(batch_features.cpu().detach())
        # features.append(batch_features)
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi


def membership_to_label(membership):
    """Turn a membership matrix into a list of labels."""
    _, num_classes, num_samples, _ = membership.shape
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        labels[i] = np.argmax(membership[:, i, i])
    return labels

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot
