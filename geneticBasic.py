# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:26:54 2020

@author: Ludwig
"""

import numpy as np
import pandas as pd

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
CUDA_LAUNCH_BLOCKING=1
print(torch.cuda.is_available())
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)



trainingDataPath ="C:/datasets/MNist/train.csv"

trainingData = pd.read_csv(trainingDataPath,dtype=np.float32)

features = trainingData.loc[:, trainingData.columns != "label"].values/255
targets = trainingData.loc[:, "label"].values

feature_train,target_train, feature_test,target_test = train_test_split(features,targets,test_size=0.2)

xTrain = torch.from_numpy(feature_train)
yTrain = torch.from_numpy(target_train)

xTest = torch.from_numpy(feature_test)
yTest = torch.from_numpy(target_test)

train = torch.utils.data.TensorDataset(xTrain,yTrain)
test = torch.utils.data.TensorDataset(xTest,yTest)

train_loader = torch.utils.data.DataLoader(train,batch_size=500,shuffle=True)
test_loader= torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)

class geneticNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.fc1 = nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,10)
        
        self.fc1.weights.requires_grad = False
        self.fc2.weights.requires_grad = False
        self.fc3.weights.requires_grad = False
        self.fc4.weights.requires_grad = False
        self.fc5.weights.requires_grad = False
        
        self.fc2.bias.requires_grad = False
        self.fc3.bias.requires_grad = False
        self.fc4.bias.requires_grad = False
        self.fc5.bias.requires_grad = False
        self.fc1.bias.requires_grad = False
        
        
        self.activation1 = F.leaky_relu
        self.activation2 = F.hardswish
        
        self.logmax = F.log_softmax
        
        self.dropout= nn.Dropout(p=0.5)
    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.dropout(self.activation1(self.fc1(x)))
        x=self.dropout(self.activation1(self.fc2(x)))
        x=self.dropout(self.activation1(self.fc3(x)))
        x=self.dropout(self.activation2(self.fc4(x)))
        
        x=self.fc5(x)
        
        return x

model = geneticNet()

sd = model.state_dict()

rand = random.Random()

accuracy = 0

targetAccuracy = 0.9

#need to save state_dict
#intialize models in list with the state dict
#randomize some of their parameters
#compare models and save the most successful



while accuracy < targetAccuracy:
    sd = model.state_dict()
    models = [geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd),geneticNet().load_state_dict(sd)]
    
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        pred = model(images)
        
        top_p, top_class = pred.topk(1,dim=1)
        
        equal = top_class == labels.view(*top_class.shape)
        
        accuracy += torch.mean(equal.type(torch.FloatTensor))
        
    accuracy /= len(train_loader)
    
    accuracies = []
    
    for candidate in models:
        acc = 0
        
        #randomize, probably best with list comprehension
        #can use rand.randn and make probability some number divided by the amount of parameters
        #for paramTensor in candidate.state_dict():
            #candidate.state_dict()[paramTensor] = [for i in candidate.state_dict()[paramTensor] for j in i ]
            
        for images,labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
        
            pred = model(images)
        
            top_p, top_class = pred.topk(1,dim=1)
        
            equal = top_class == labels.view(*top_class.shape)
        
            acc += torch.mean(equal.type(torch.FloatTensor))
        
        acc /= train_loader
        accuracies.append(acc)
        
    candidateAccuracy = max(accuracies)
    index = accuracies.index(candidateAccuracy)
    
    if candidateAccuracy>accuracy:
        model=models[index]
        
    
    



