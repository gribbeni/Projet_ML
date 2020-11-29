#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler
import matplotlib.pyplot as plt
import os
from functools import reduce
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sklearn 


def make_simplers(train_data,valid_size):
    
    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

    train_sampler = ImbalancedDatasetSampler(train_data,indices=train_new_idx)
    valid_sampler = ImbalancedDatasetSampler(train_data,indices=valid_idx)
    
    return train_sampler,valid_sampler


def make_train_val_loader(train_dir,transform,valid_size,batch_size) : 
    
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    
    train_sampler,valid_sampler = make_simplers(train_data, valid_size)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
    
    return train_loader,valid_loader


def make_test_loader(test_dir,transform,batch_size):
    
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    
    return test_loader


def make_all_loaders(train_dir,test_dir,transform,valid_size,batch_size):
    
    train_loader,valid_loader = make_train_val_loader(train_dir,transform,valid_size,batch_size)
    
    test_loader=make_test_loader(test_dir,transform,batch_size)
    
    return train_loader,valid_loader,test_loader







