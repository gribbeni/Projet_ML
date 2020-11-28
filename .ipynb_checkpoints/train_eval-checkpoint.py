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
from tqdm import tqdm

# In[2]:


import loaders as l
import models as m


# In[6]:


def calc_metrics_v1(net,loader,show,device) : 

    all_labels=[]
    all_predicted=[]
    
    for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            #data
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


# In[9]:


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
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch[0].to(device), batch[1].to(device)
            #batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            #all_labels.extend(np.array(labels))

            _, predicted = torch.max(outputs.data, 1)
            #all_predicted.extend(np.array(predicted))

            running_loss += loss.item()
            if n_batch % 2000 == 1999:    # print every 2000 mini-batches

                #peut-être ça ne sert plus---------
                y_pred_softmax = torch.log_softmax(outputs, dim = 1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                outputs=np.array(y_pred_tags.cpu())
                labels=np.array(labels.cpu())
                #---------------------------------
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


# In[ ]:





# In[8]:



'''
train_dir = 'C:/Users/33783/Desktop/start_deep/start_deep/train_images'
test_dir = 'C:/Users/33783/Desktop/start_deep/start_deep/test_images'

transform = transforms.Compose(
    [transforms.Grayscale(), 
     transforms.ToTensor(), 
     transforms.Normalize(mean=(0,),std=(1,))])

valid_size = 0.2
batch_size = 32

train_loader,valid_loader,test_loader=l.make_all_loaders(train_dir,test_dir,transform,valid_size,batch_size)
classes = ('noface','face')

net = m.BaseNet()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs=2
#Training

all_labels,all_predicted,all_losses,all_accuracies,all_f1scores= train_v1(net,criterion,optimizer,epochs,train_loader,valid_loader)
            
plt.plot(all_losses, color='blue')
plt.plot(all_f1scores, color='red')
'''


# In[ ]:





# In[ ]:




