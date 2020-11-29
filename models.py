#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torchvision
import os
from torch.optim import Adam
from functools import reduce
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sklearn 

# In[ ]:


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = reduce(lambda r,s: r*s,size)
        #print(num_features)
        return num_features


# In[ ]:

class BaseNet_v2(nn.Module):
    def __init__(self):
        super(BaseNet_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = reduce(lambda r,s: r*s,size)
        #print(num_features)
        return num_features



# In[ ]:
class BaseNet_v3(nn.Module):
    def __init__(self):
        super(BaseNet_v3, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 20, 3)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4500, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = reduce(lambda r,s: r*s,size)
        #print(num_features)
        return num_features



# In[ ]:


class BaseNet_v4(nn.Module):
    def __init__(self):
        super(BaseNet_v4, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 20, 3)
        self.conv3 = nn.Conv2d(20, 18, 3)
        self.conv4 = nn.Conv2d(18, 16, 3)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = reduce(lambda r,s: r*s,size)
        #print(num_features)
        return num_features

    
class BaseNet_v5(nn.Module):
    def __init__(self):
        super(BaseNet_v5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5,padding=2,stride=2)
        self.conv2 = nn.Conv2d(32,64 , 3,padding=2,stride=2)
        self.conv3 = nn.Conv2d(64,128 , 3,padding=2,stride=2)
        #self.dropout = nn.Dropout(0.2)
        self.aap  = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
 
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #x = self.dropout(x)
        x = self.aap(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = reduce(lambda r,s: r*s,size)
        #print(num_features)
        return num_features
