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

test_LR = torchvision.datasets.ImageFolder(root='2_LR_test', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)

def psnr(target, ref):
    # target:目标图像  ref:参考图像 
    # assume RGB image
    diff = ref- target
    diff = diff.reshape(-1)
    rmse = torch.sqrt((diff ** 2.).mean())
    return 20*torch.log10(1.0/rmse)

HR_URL='2_HR'
LR_URL='2_LR'
        
        
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
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
        
        
        
        
net1=Net1()
net1.load_state_dict(torch.load('real_net1.pt'))
net1.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net1.to(device)
        
criterion = nn.L1Loss()
optimizer=torch.optim.Adam(net1.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for i in range(20):
    LR_set=torchvision.datasets.ImageFolder(root=LR_URL,transform=transforms.ToTensor())
    LR_loader=torch.utils.data.DataLoader(LR_set,batch_size=1,shuffle=False,num_workers=0)
    LR_loader=iter(LR_loader)
    HR_set=torchvision.datasets.ImageFolder(root=HR_URL,transform=transforms.ToTensor())
    HR_loader=torch.utils.data.DataLoader(HR_set,batch_size=1,shuffle=False,num_workers=0)
    HR_loader=iter(HR_loader)
    for j in range(len(LR_set)):
        LR=LR_loader.next()[0]
        HR=HR_loader.next()[0]
        _,_,x,y=LR.size()
        n=4
        LR=torch.nn.functional.unfold(LR,(int(x/n),int(y/n)),stride=(int(x/n),int(y/n)))
        HR=torch.nn.functional.unfold(HR,(int(x/n),int(y/n)),stride=(int(x/n),int(y/n)))
        for k in range(n*n):
            optimizer.zero_grad()
            LR_temp=LR[:,:,k].reshape([1,3,int(x/n),int(y/n)]).to(device)
            LR_temp=net1(LR_temp)
            HR_temp=HR[:,:,k].reshape([1,3,int(x/n),int(y/n)]).to(device)
            Loss=criterion(LR_temp,HR_temp)
            Loss.backward()
            optimizer.step()
            
        del LR,HR,LR_temp,HR_temp
        print('[%d, %5d] loss: %.3f' %(i + 1, j + 1, Loss.item()))

    n_test=15
    PSNR=0.0
    temp_HR_2=iter(HR_2_test)
    temp_LR_2=iter(LR_2_test)
    for t in range(n_test):
        with torch.no_grad():
            HR_test=temp_HR_2.next()[0].squeeze()
            _,x,y=HR_test.size()
            LR_test=temp_LR_2.next()[0]
            if x>1500:
                x=1500
            if y>1500:
                y=1500
            HR_test=HR_test[:,0:x,0:y].to(device)
            LR_test=LR_test[:,:,0:x,0:y].to(device)
            outputs=net1(LR_test).data.squeeze()
            PSNR+=psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
    torch.save(net1.state_dict(), 'real_net1.pt')
        
        

        

        
        
        
        
        
        
        
        
