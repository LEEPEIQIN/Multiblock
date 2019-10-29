#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from random import randrange
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import math
import os

class Net2_plus(nn.Module):
    def __init__(self):
        super(Net2_plus, self).__init__()
        self.pooling=nn.AdaptiveAvgPool2d(1)
        #initial cnn:
        self.int = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        #RB_1:
        self.RB_1_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_1_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_1_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_1_1d2 = nn.Conv1d(16,64,1,1)
        #RB_2:
        self.RB_2_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_2_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_2_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_2_1d2 = nn.Conv1d(16,64,1,1)
        #RB_3:
        self.RB_3_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_3_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_3_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_3_1d2 = nn.Conv1d(16,64,1,1)
        #RB_4:
        self.RB_4_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_4_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_4_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_4_1d2 = nn.Conv1d(16,64,1,1)
        #RB_5:
        self.RB_5_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_5_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_5_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_5_1d2 = nn.Conv1d(16,64,1,1)
        #RB_6:
        self.RB_6_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_6_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_6_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_6_1d2 = nn.Conv1d(16,64,1,1)
        #RB_7:
        self.RB_7_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_7_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_7_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_7_1d2 = nn.Conv1d(16,64,1,1)
        #RB_8:
        self.RB_8_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_8_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_8_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_8_1d2 = nn.Conv1d(16,64,1,1)
        #RB_9:
        self.RB_9_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_9_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_9_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_9_1d2 = nn.Conv1d(16,64,1,1)
        #RB_10:
        self.RB_10_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_10_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_10_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_10_1d2 = nn.Conv1d(16,64,1,1)
         #RB_11:
        self.RB_11_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_11_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_11_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_11_1d2 = nn.Conv1d(16,64,1,1)
        #RB_12:
        self.RB_12_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_12_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_12_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_12_1d2 = nn.Conv1d(16,64,1,1)
        #RB_13:
        self.RB_13_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_13_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_13_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_13_1d2 = nn.Conv1d(16,64,1,1)
        #RB_14:
        self.RB_14_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_14_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_14_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_14_1d2 = nn.Conv1d(16,64,1,1)
        #RB_15:
        self.RB_15_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_15_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_15_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_15_1d2 = nn.Conv1d(16,64,1,1)
        #RB_16:
        self.RB_16_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_16_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RB_16_1d1 = nn.Conv1d(64,16,1,1)
        self.RB_16_1d2 = nn.Conv1d(16,64,1,1)
        #RB_17:
        #self.RB_17_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_17_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_17_1d1 = nn.Conv1d(64,16,1,1)
        #self.RB_17_1d2 = nn.Conv1d(16,64,1,1)
        #RB_18:
        #self.RB_18_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_18_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_18_1d1 = nn.Conv1d(64,16,1,1)
        #self.RB_18_1d2 = nn.Conv1d(16,64,1,1)
        #RB_19:
        #self.RB_19_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_19_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_19_1d1 = nn.Conv1d(64,16,1,1)
        #self.RB_19_1d2 = nn.Conv1d(16,64,1,1)
        #RB_20:
        #self.RB_20_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_20_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        #self.RB_20_1d1 = nn.Conv1d(64,16,1,1)
        #self.RB_20_1d2 = nn.Conv1d(16,64,1,1)
        #end cnn:
        self.end_RB=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.toR=nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
       
        self._initialize_weights()
    def forward(self, x):
        LR=x
        x=self.int(x)
        
        #RB_1
        R_1=x
        x=F.relu(self.RB_1_conv1(x))
        x=self.RB_1_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_1_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_1_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_1
        #RB_2
        R_2=x
        x=F.relu(self.RB_2_conv1(x))
        x=self.RB_2_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_2_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_2_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_2
        #RB_3
        R_3=x
        x=F.relu(self.RB_3_conv1(x))
        x=self.RB_3_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_3_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_3_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_3
        #RB_4
        R_4=x
        x=F.relu(self.RB_4_conv1(x))
        x=self.RB_4_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_4_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_4_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_4
        #RB_5
        R_5=x
        x=F.relu(self.RB_5_conv1(x))
        x=self.RB_5_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_5_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_5_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_5
        #RB_6
        R_6=x
        x=F.relu(self.RB_6_conv1(x))
        x=self.RB_6_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_6_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_6_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_6
        #RB_7
        R_7=x
        x=F.relu(self.RB_7_conv1(x))
        x=self.RB_7_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_7_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_7_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_7
        #RB_8
        R_8=x
        x=F.relu(self.RB_8_conv1(x))
        x=self.RB_8_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_8_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_8_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_8
        #RB_9
        R_9=x
        x=F.relu(self.RB_9_conv1(x))
        x=self.RB_9_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_9_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_9_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_9
        #RB_10
        R_10=x
        x=F.relu(self.RB_10_conv1(x))
        x=self.RB_10_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_10_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_10_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_10
        
        #RB_11
        R_11=x
        x=F.relu(self.RB_11_conv1(x))
        x=self.RB_11_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_11_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_11_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_11
        #RB_12
        R_12=x
        x=F.relu(self.RB_12_conv1(x))
        x=self.RB_12_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_12_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_12_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_12
        #RB_13
        R_13=x
        x=F.relu(self.RB_13_conv1(x))
        x=self.RB_13_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_13_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_13_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_13
        #RB_14
        R_14=x
        x=F.relu(self.RB_14_conv1(x))
        x=self.RB_14_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_14_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_14_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_14
        #RB_15
        R_15=x
        x=F.relu(self.RB_15_conv1(x))
        x=self.RB_15_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_15_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_15_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_15
        #RB_16
        R_16=x
        x=F.relu(self.RB_16_conv1(x))
        x=self.RB_16_conv2(x)
        ca=self.pooling(x)
        ca=F.relu(self.RB_16_1d1(ca.squeeze(2)))
        ca=torch.sigmoid(self.RB_16_1d2(ca))
        x=x*ca.unsqueeze(3)
        x+=R_16
        #RB_17
        #R_17=x
        #x=F.relu(self.RB_17_conv1(x))
        #x=self.RB_17_conv2(x)
        #ca=self.pooling(x)
        #ca=F.relu(self.RB_17_1d1(ca.squeeze(2)))
        #ca=torch.sigmoid(self.RB_17_1d2(ca))
        #x=x*ca.unsqueeze(3)
        #x+=R_17
        #RB_18
        #R_18=x
        #x=F.relu(self.RB_18_conv1(x))
        #x=self.RB_18_conv2(x)
        #ca=self.pooling(x)
        #ca=F.relu(self.RB_18_1d1(ca.squeeze(2)))
        #ca=torch.sigmoid(self.RB_18_1d2(ca))
        #x=x*ca.unsqueeze(3)
        #x+=R_18
        #RB_19
        #R_19=x
        #x=F.relu(self.RB_19_conv1(x))
        #x=self.RB_19_conv2(x)
        #ca=self.pooling(x)
        #ca=F.relu(self.RB_19_1d1(ca.squeeze(2)))
        #ca=torch.sigmoid(self.RB_19_1d2(ca))
        #x=x*ca.unsqueeze(3)
        #x+=R_19
        #RB_20
        #R_20=x
        #x=F.relu(self.RB_20_conv1(x))
        #x=self.RB_20_conv2(x)
        #ca=self.pooling(x)
        #ca=F.relu(self.RB_20_1d1(ca.squeeze(2)))
        #ca=torch.sigmoid(self.RB_20_1d2(ca))
        #x=x*ca.unsqueeze(3)
        #x+=R_20
        
        #end:
        x=self.end_RB(x)
        x=self.toR(x)
        x+=LR
        x[x < 0.0] = 0.0
        x[x>1.0]=1.0
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.int.weight, init.calculate_gain('relu'))
        
        #RB_1
        init.orthogonal_(self.RB_1_conv1.weight)
        init.orthogonal_(self.RB_1_conv2.weight)
        init.orthogonal_(self.RB_1_1d1.weight)
        init.orthogonal_(self.RB_1_1d2.weight)
        #RB_2
        init.orthogonal_(self.RB_2_conv1.weight)
        init.orthogonal_(self.RB_2_conv2.weight)
        init.orthogonal_(self.RB_2_1d1.weight)
        init.orthogonal_(self.RB_2_1d2.weight)
        #RB_3
        init.orthogonal_(self.RB_3_conv1.weight)
        init.orthogonal_(self.RB_3_conv2.weight)
        init.orthogonal_(self.RB_3_1d1.weight)
        init.orthogonal_(self.RB_3_1d2.weight)
        #RB_4
        init.orthogonal_(self.RB_4_conv1.weight)
        init.orthogonal_(self.RB_4_conv2.weight)
        init.orthogonal_(self.RB_4_1d1.weight)
        init.orthogonal_(self.RB_4_1d2.weight)
        #RB_5
        init.orthogonal_(self.RB_5_conv1.weight)
        init.orthogonal_(self.RB_5_conv2.weight)
        init.orthogonal_(self.RB_5_1d1.weight)
        init.orthogonal_(self.RB_5_1d2.weight)
        #RB_6
        init.orthogonal_(self.RB_6_conv1.weight)
        init.orthogonal_(self.RB_6_conv2.weight)
        init.orthogonal_(self.RB_6_1d1.weight)
        init.orthogonal_(self.RB_6_1d2.weight)
        #RB_7
        init.orthogonal_(self.RB_7_conv1.weight)
        init.orthogonal_(self.RB_7_conv2.weight)
        init.orthogonal_(self.RB_7_1d1.weight)
        init.orthogonal_(self.RB_7_1d2.weight)
        #RB_8
        init.orthogonal_(self.RB_8_conv1.weight)
        init.orthogonal_(self.RB_8_conv2.weight)
        init.orthogonal_(self.RB_8_1d1.weight)
        init.orthogonal_(self.RB_8_1d2.weight)
        #RB_9
        init.orthogonal_(self.RB_9_conv1.weight)
        init.orthogonal_(self.RB_9_conv2.weight)
        init.orthogonal_(self.RB_9_1d1.weight)
        init.orthogonal_(self.RB_9_1d2.weight)
        #RB_10
        init.orthogonal_(self.RB_10_conv1.weight)
        init.orthogonal_(self.RB_10_conv2.weight)
        init.orthogonal_(self.RB_10_1d1.weight)
        init.orthogonal_(self.RB_10_1d2.weight)
        #RB_11
        init.orthogonal_(self.RB_11_conv1.weight)
        init.orthogonal_(self.RB_11_conv2.weight)
        init.orthogonal_(self.RB_11_1d1.weight)
        init.orthogonal_(self.RB_11_1d2.weight)
        #RB_12
        init.orthogonal_(self.RB_12_conv1.weight)
        init.orthogonal_(self.RB_12_conv2.weight)
        init.orthogonal_(self.RB_12_1d1.weight)
        init.orthogonal_(self.RB_12_1d2.weight)
        #RB_13
        init.orthogonal_(self.RB_13_conv1.weight)
        init.orthogonal_(self.RB_13_conv2.weight)
        init.orthogonal_(self.RB_13_1d1.weight)
        init.orthogonal_(self.RB_13_1d2.weight)
        #RB_14
        init.orthogonal_(self.RB_14_conv1.weight)
        init.orthogonal_(self.RB_14_conv2.weight)
        init.orthogonal_(self.RB_14_1d1.weight)
        init.orthogonal_(self.RB_14_1d2.weight)
        #RB_15
        init.orthogonal_(self.RB_15_conv1.weight)
        init.orthogonal_(self.RB_15_conv2.weight)
        init.orthogonal_(self.RB_15_1d1.weight)
        init.orthogonal_(self.RB_15_1d2.weight)
        #RB_16
        init.orthogonal_(self.RB_16_conv1.weight)
        init.orthogonal_(self.RB_16_conv2.weight)
        init.orthogonal_(self.RB_16_1d1.weight)
        init.orthogonal_(self.RB_16_1d2.weight)
        #RB_17
        #init.orthogonal_(self.RB_17_conv1.weight)
        #init.orthogonal_(self.RB_17_conv2.weight)
        #init.orthogonal_(self.RB_17_1d1.weight)
        #init.orthogonal_(self.RB_17_1d2.weight)
        #RB_18
        #init.orthogonal_(self.RB_18_conv1.weight)
        #init.orthogonal_(self.RB_18_conv2.weight)
        #init.orthogonal_(self.RB_18_1d1.weight)
        #init.orthogonal_(self.RB_18_1d2.weight)
        #RB_19
        #init.orthogonal_(self.RB_19_conv1.weight)
        #init.orthogonal_(self.RB_19_conv2.weight)
        #init.orthogonal_(self.RB_19_1d1.weight)
        #init.orthogonal_(self.RB_19_1d2.weight)
        #RB_20
        #init.orthogonal_(self.RB_20_conv1.weight)
        #init.orthogonal_(self.RB_20_conv2.weight)
        #init.orthogonal_(self.RB_20_1d1.weight)
        #init.orthogonal_(self.RB_20_1d2.weight)
        
        init.orthogonal_(self.end_RB.weight)
        init.orthogonal_(self.toR.weight)
        
        
        
        
net2_plus=Net2_plus()
net2_plus.load_state_dict(torch.load('real_net2_plus.pt'))
net2_plus.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net2_plus.to(device)

cwd = os.getcwd()

#####################################################################3
LR_URL='2_LR_afternet1'

LR_set=torchvision.datasets.ImageFolder(root=LR_URL,transform=transforms.ToTensor())
LR_loader=torch.utils.data.DataLoader(LR_set,batch_size=1,shuffle=False,num_workers=0)
LR_loader=iter(LR_loader)

os.mkdir('2_LR_afternet2')
os.chdir('2_LR_afternet2')
os.mkdir('2')
os.chdir('2')

wd_1=os.getcwd()
os.chdir(cwd)

with torch.no_grad():
    for i in range(len(LR_set)):
        LR=LR_loader.next()[0]
        _,_,x,y=LR.size()
        temp=torch.nn.functional.unfold(LR,(int(x/2),int(y/2)),stride=(int(x/2),int(y/2)))
        LR_tl=temp[:,:,0].reshape([1,3,int(x/2),int(y/2)]).to(device)
        LR_tr=temp[:,:,1].reshape([1,3,int(x/2),int(y/2)]).to(device)
        LR_bl=temp[:,:,2].reshape([1,3,int(x/2),int(y/2)]).to(device)
        LR_br=temp[:,:,3].reshape([1,3,int(x/2),int(y/2)]).to(device)
        LR_1=net2_plus(LR_tl).data
        LR_2=net2_plus(LR_tr).data
        LR_3=net2_plus(LR_bl).data
        LR_4=net2_plus(LR_br).data
        temp1=torch.cat((LR_1,LR_2),3)
        temp2=torch.cat((LR_3,LR_4),3)
        temp=torch.cat((temp1,temp2),2).squeeze()
        LR_after=temp.cpu()
        LR_after = transforms.ToPILImage()(LR_after)
        os.chdir(wd_1)
        if i<9:
            LR_after.save("2_LR_afternet2_00"+str(i+1)+".png","PNG")
        else:
            LR_after.save("2_LR_afternet2_0"+str(i+1)+".png","PNG")
        os.chdir(cwd)
        print('done')
        
##
        
LR_URL='2_LR_test_afternet1'

LR_set=torchvision.datasets.ImageFolder(root=LR_URL,transform=transforms.ToTensor())
LR_loader=torch.utils.data.DataLoader(LR_set,batch_size=1,shuffle=False,num_workers=0)
LR_loader=iter(LR_loader)

os.mkdir('2_LR_test_afternet2')
os.chdir('2_LR_test_afternet2')
os.mkdir('2')
os.chdir('2')

wd_2=os.getcwd()
os.chdir(cwd)
        
with torch.no_grad():
   for i in range(len(LR_set)):
       LR=LR_loader.next()[0]
       _,_,x,y=LR.size()
       temp=torch.nn.functional.unfold(LR,(int(x/2),int(y/2)),stride=(int(x/2),int(y/2)))
       LR_tl=temp[:,:,0].reshape([1,3,int(x/2),int(y/2)]).to(device)
       LR_tr=temp[:,:,1].reshape([1,3,int(x/2),int(y/2)]).to(device)
       LR_bl=temp[:,:,2].reshape([1,3,int(x/2),int(y/2)]).to(device)
       LR_br=temp[:,:,3].reshape([1,3,int(x/2),int(y/2)]).to(device)
       LR_1=net2_plus(LR_tl).data
       LR_2=net2_plus(LR_tr).data
       LR_3=net2_plus(LR_bl).data
       LR_4=net2_plus(LR_br).data
       temp1=torch.cat((LR_1,LR_2),3)
       temp2=torch.cat((LR_3,LR_4),3)
       temp=torch.cat((temp1,temp2),2).squeeze()
       LR_after=temp.cpu()
       LR_after = transforms.ToPILImage()(LR_after)
       os.chdir(wd_2)
       if i<9:
            LR_after.save("2_LR_test_afternet2_00"+str(i+1)+".png","PNG")
       else:
            LR_after.save("2_LR_test_afternet2_0"+str(i+1)+".png","PNG")
       os.chdir(cwd)
       print('done')