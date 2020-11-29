#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler
import matplotlib.pyplot as plt
import os
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from functools import reduce
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pandas as pd
import sklearn 
import loaders as l
import models as m


def calc_metrics_v1(net,loader,show,device) : 

    all_labels=[]
    all_predicted=[]
    
    for i, data in enumerate(loader, 0):
            
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            
            y_pred_softmax = torch.log_softmax(outputs, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
            
            all_predicted.extend(np.array(y_pred_tags.cpu()))
            all_labels.extend(np.array(labels.cpu()))
    
    f1s=f1_score(all_labels,all_predicted,average='weighted')
    prec=precision_score(all_labels,all_predicted,average='weighted')
    rec=recall_score(all_labels,all_predicted,average='weighted')
    cf_m=pd.DataFrame(sklearn.metrics.confusion_matrix(all_labels, all_predicted,normalize='true'))
    roc_aucc=roc_auc_score(all_labels,all_predicted)
    
    if show ==True : 
        print('Validation set')
        print("f1_score ",f1s)
        print("precision ",prec)
        print("recall ",rec)
        print("confusion matrix\n", cf_m)
        
    return f1s,prec,rec,cf_m,roc_aucc




def train_v1(net,criterion,optimizer,epochs,train_loader,valid_loader,device):
    
    all_labels=[]
    all_predicted=[]
    all_losses=[]
    all_accuracies=[]
    all_f1scores=[]
    all_roc_auc=[]
    bestf1=0
    best_params=net.state_dict()
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for n_batch,batch in enumerate(train_loader) :
            
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if n_batch % 2000 == 1999:    # print every 2000 mini-batches so 1 epoch

                all_losses.append(running_loss / n_batch)
                f1,_,_,_,roc = calc_metrics_v1(net,valid_loader,False,device)#[0]
                
                if f1 > bestf1 :
                    bestf1=f1
                    best_params=net.state_dict()
                    
                all_f1scores.append(f1)
                all_roc_auc.append(roc)
                
                print('[%2d, %2d] loss: %.3f f1_score on validation set : %.3f' %
                      (epoch + 1, n_batch + 1, running_loss / n_batch,f1))

                running_loss = 0.0
            
    
    return all_losses,all_accuracies,all_f1scores,all_roc_auc,best_params




