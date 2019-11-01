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
 
 
def psnr(target, ref):
    # target:目标图像  ref:参考图像 
    # assume RGB image
    target_data = np.array(target)
    ref_data = np.array(ref)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)

#test loader:
test_LR = torchvision.datasets.ImageFolder(root='/home/lee/Desktop/python/sillyman/2_LR_test', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='/home/lee/Desktop/python/sillyman/2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)
## train set generator:
def generator():
    #given:
    HR_URL='/home/lee/Desktop/python/sillyman/2_HR'
    LR_URL='/home/lee/Desktop/python/sillyman/2_LR'
    batch_size=64
    image_size=61
    channel=3
    #genereting:
    HR_set=torchvision.datasets.ImageFolder(root=HR_URL)
    LR_set=torchvision.datasets.ImageFolder(root=LR_URL)
    HR_torch=torch.zeros([batch_size,channel,image_size,image_size])
    LR_torch=torch.zeros([batch_size,channel,image_size,image_size])
    for i in range(batch_size):
        img_number=randrange(len(HR_set))
        HR_img=Image.open(HR_set.imgs[img_number][0])
        LR_img=Image.open(LR_set.imgs[img_number][0])
        y,x=HR_img.size
        X=randrange(0,x-image_size)
        Y=randrange(0,y-image_size)
        HR_img=torchvision.transforms.functional.crop(HR_img, X, Y, image_size, image_size)
        LR_img=torchvision.transforms.functional.crop(LR_img, X, Y, image_size, image_size)
        #whether rotate and whether flipHR_:
        p_f=randrange(2)
        d_r=randrange(4)*90
        HR_img=torchvision.transforms.functional.rotate(HR_img,d_r)
        LR_img=torchvision.transforms.functional.rotate(LR_img,d_r)
        if (p_f==1):
            HR_img=HR_img.transpose(Image.FLIP_LEFT_RIGHT)
            LR_img=LR_img.transpose(Image.FLIP_LEFT_RIGHT)
        ##tensor defined:
        HR_torch[i,:,:,:]=torchvision.transforms.functional.to_tensor(HR_img)
        LR_torch[i,:,:,:]=torchvision.transforms.functional.to_tensor(LR_img)
    ans=[HR_torch,LR_torch]
    return ans
#######
    

##net work define:
class RE_12conv(nn.Module):
    def __init__(self):
        super(RE_12conv, self).__init__()
        
        self.int = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv9 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv10 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.end = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        
        self._initialize_weights()

    def forward(self, x):
        keep=x
        x=self.int(x)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=(self.conv5(x))
        x+=keep
        keep=x
        x=(self.conv6(x))
        x=F.relu(self.conv7(x))
        x=F.relu(self.conv8(x))
        x=F.relu(self.conv9(x))
        x=F.relu(self.conv10(x))
        x=self.end(x)
        x+=keep
        return x

    def _initialize_weights(self):
        init.normal_(self.int.weight, mean=0.0, std=0.1)
        init.normal_(self.conv1.weight, mean=0.0, std=0.1)
        init.normal_(self.conv2.weight, mean=0.0, std=0.1)
        init.normal_(self.conv3.weight, mean=0.0, std=0.1)
        init.normal_(self.conv4.weight, mean=0.0, std=0.1)
        init.normal_(self.conv5.weight, mean=0.0, std=0.1)
        init.normal_(self.conv6.weight, mean=0.0, std=0.1)
        init.normal_(self.conv7.weight, mean=0.0, std=0.1)
        init.normal_(self.conv8.weight, mean=0.0, std=0.1)
        init.normal_(self.conv9.weight, mean=0.0, std=0.1)
        init.normal_(self.conv10.weight, mean=0.0, std=0.1)
        init.normal_(self.end.weight, mean=0.0, std=0.1)
        
residual_12conv=RE_12conv()
#########################
criterion = nn.L1Loss()
optimizer=torch.optim.Adam(residual_12conv.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
 
##start train:
p=[]
for epoch in range(20):
    running_loss=0.0
    for i in range(50):
        HR,LR=generator()
        optimizer.zero_grad()
        outputs=residual_12conv(LR)
        Loss=criterion(outputs,HR)
        Loss.backward()
        optimizer.step()
        running_loss+=Loss.item()
        
        print('[%d, %5d] loss: %.3f' %
             (epoch + 1, i + 1, running_loss))
        running_loss = 0.0
        del HR,LR,Loss,outputs
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
            HR_test=HR_test[:,0:x,0:y]
            LR_test=LR_test[:,:,0:x,0:y]
            outputs=residual_12conv(LR_test).squeeze()
            PSNR+=psnr(outputs.data,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    p.append(PSNR)
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
print('Finished Training')

optimizer=torch.optim.Adam(residual_12conv.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
p=[]
for epoch in range(10):
    running_loss=0.0
    for i in range(50):
        HR,LR=generator()
        optimizer.zero_grad()
        outputs=residual_12conv(LR)
        Loss=criterion(outputs,HR)
        Loss.backward()
        optimizer.step()
        running_loss+=Loss.item()
        
        print('[%d, %5d] loss: %.3f' %
             (epoch + 1, i + 1, running_loss))
        running_loss = 0.0
        del HR,LR,Loss,outputs
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
            HR_test=HR_test[:,0:x,0:y]
            LR_test=LR_test[:,:,0:x,0:y]
            outputs=residual_12conv(LR_test).squeeze()
            PSNR+=psnr(outputs.data,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    p.append(PSNR)
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
print('Finished Training')
