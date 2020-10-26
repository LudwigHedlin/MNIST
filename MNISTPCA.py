# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

#PCA analysis
n_components = 100
pca = PCA(n_components)
x_= pca.fit_transform(x_)

feature_train,feature_test,target_train,target_test = train_test_split(x_,y_,test_size=0.2)

x_train = torch.from_numpy(feature_train)
y_train = torch.from_numpy(target_train).type(torch.LongTensor)


x_test = torch.from_numpy(feature_test)
y_test = torch.from_numpy(target_test).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(train,batch_size=256,shuffle=True)
test_loader= torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)

class Trainer(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(n_components,1000)
        self.layer2 = nn.Linear(1000,1200)
        self.layer3 = nn.Linear(1200,1000)
        self.layer4 = nn.Linear(1000,800)
        self.layer5 = nn.Linear(800,700)
        self.layer6 = nn.Linear(700,500)
        self.layer7 = nn.Linear(500,350)
        self.layer8 = nn.Linear(350,200)
        self.layer9 = nn.Linear(200,100)
        self.layer10 = nn.Linear(100,64)
        self.layer11 = nn.Linear(64,10)
        
        self.dropout = nn.Dropout(p=0.44)
        
        self.logmax = F.log_softmax
        
    def forward(self,x):
        x = self.dropout(F.leaky_relu(self.layer1(x)))
        x = self.dropout(F.leaky_relu(self.layer2(x)))
        x = self.dropout(F.leaky_relu(self.layer3(x)))
        x = self.dropout(F.leaky_relu(self.layer4(x)))
        x = self.dropout(F.leaky_relu(self.layer5(x)))
        x = self.dropout(F.leaky_relu(self.layer6(x)))
        x = self.dropout(F.leaky_relu(self.layer7(x)))
        x = self.dropout(F.leaky_relu(self.layer8(x)))
        x = self.dropout(F.leaky_relu(self.layer9(x)))
        x = self.dropout(F.leaky_relu(self.layer10(x)))
        
        x = self.logmax(self.layer11(x),dim=1)
        
        return x
    
    
model = Trainer()
model.to(device)

criterion = nn.NLLLoss()
criterion.to(device)

rate=0.0005
optimizer = optim.Adam(model.parameters(),lr=rate)

epoch=20
steps = 0
print_every = 50
train_losses, test_losses = [],[]

for e in range(epoch):
    running_loss = 0
    
    for images,labels in train_loader:
        
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
        