# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:59:49 2020

@author: Ludwig
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt

#from pathlib import Path


#importing the dataset 
CUDA_LAUNCH_BLOCKING=1
print(torch.cuda.is_available())
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)

df = pd.read_csv("C:/Datasets/MNist/train.csv",dtype=np.float32)

x_ = df.loc[:,df.columns != "label"].values/255
y_ = df.loc[:,"label"].values

feature_train,feature_test,target_train,target_test = train_test_split(x_,y_,test_size=0.2)

x_train = torch.from_numpy(feature_train)
y_train = torch.from_numpy(target_train).type(torch.LongTensor)

x_train = x_train.reshape(33600,28,28)


x_test = torch.from_numpy(feature_test)
y_test = torch.from_numpy(target_test).type(torch.LongTensor)

x_test = x_test.reshape(8400,28,28)

train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(train,batch_size=256,shuffle=True)
test_loader= torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        """The convolution expands the channels to 3 from 1 (grayscale) and removes two rows/columns.
        For example a 28*28 grayscale picture (as in MNIST) becomes a 3-channel 26*26 picture. In general"""
        self.conv1 = nn.Conv2d(1, 3, 5) 
        self.pool = nn.MaxPool2d(2,2) #This halves the pixel dimensions.
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.conv3 = nn.Conv2d(6, 10, 3)
        self.fc1 = nn.Linear(10*10*10,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,10)
        
        self.activation1 = F.leaky_relu
        self.activation2 = F.hardswish
        
        self.logmax = F.log_softmax
        
        self.dropout= nn.Dropout(p=0.5)
    def forward(self,x):
        x=F.leaky_relu(self.conv1(x))
        x=F.leaky_relu(self.conv2(x))
        x=self.pool(F.leaky_relu(self.conv3(x)))
        x=x.view(-1,10*10*10)
        x=self.dropout(self.activation1(self.fc1(x)))
        x=self.dropout(self.activation1(self.fc2(x)))
        x=self.dropout(self.activation1(self.fc3(x)))
        x=self.dropout(self.activation2(self.fc4(x)))
        
        x=self.logmax(self.fc5(x),dim=1)
        
        return x
    
model = ConvNet()
model.to(device)

criterion = nn.NLLLoss()
criterion.to(device)

rate=0.01
optimizer = optim.Adam(model.parameters(),lr=rate)

epoch=10
steps = 0
print_every = 50
train_losses, test_losses = [],[]

for e in range(epoch):
    running_loss = 0
    
    for images,labels in train_loader:
        images=images.reshape(-1,1,28,28)
        
        images = images.to(device)
        labels = labels.to(device)
        
        steps += 1
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps,labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                
                model.eval()
                
                for images, labels in test_loader:
                    images= images.view(-1,1,28,28)
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    
                    log_ps = model(images)
                    test_loss += criterion(log_ps,labels)
                    ps = torch.exp(log_ps)
                    
                    top_p, top_class = ps.topk(1,dim=1)
                    
                    equal = top_class == labels.view(*top_class.shape)
                    
                    accuracy += torch.mean(equal.type(torch.FloatTensor))
                
                model.train()
            
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))
            
            
            print("Epoch : {} / {}".format(e+1,epoch),
                 "train_loss : {:.3f}".format(train_losses[-1]),
                 "test_loss : {:.3f}".format(test_losses[-1]),
                 "test accuracy : {:.3f}".format(accuracy / len(test_loader)))
            
rate=0.001
model.dropout=nn.Dropout(p=0.3)
optimizer = optim.Adam(model.parameters(),lr=rate)

epoch=10
        
for e in range(epoch):
    running_loss = 0
    
    for images,labels in train_loader:
        images=images.reshape(-1,1,28,28)
        
        images = images.to(device)
        labels = labels.to(device)
        
        steps += 1
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps,labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                
                model.eval()
                
                for images, labels in test_loader:
                    images= images.view(-1,1,28,28)
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    
                    log_ps = model(images)
                    test_loss += criterion(log_ps,labels)
                    ps = torch.exp(log_ps)
                    
                    top_p, top_class = ps.topk(1,dim=1)
                    
                    equal = top_class == labels.view(*top_class.shape)
                    
                    accuracy += torch.mean(equal.type(torch.FloatTensor))
                
                model.train()
            
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))
            
            
            print("Epoch : {} / {}".format(e+1,epoch),
                 "train_loss : {:.3f}".format(train_losses[-1]),
                 "test_loss : {:.3f}".format(test_losses[-1]),
                 "test accuracy : {:.3f}".format(accuracy / len(test_loader)))  
        
