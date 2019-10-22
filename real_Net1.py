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
test_LR = torchvision.datasets.ImageFolder(root='2_LR_test', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)
## train set generator:
def generator():
    #given:
    HR_URL='2_HR'
    LR_URL='2_LR'
    batch_size=64
    image_size=101
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
        
        #end cnn:
        self.end_RB=nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.toR=nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
       

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
        #end:
        x=self.end_RB(x)
        x=self.toR(x)
        x+=LR
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
        
        init.orthogonal_(self.end_RB.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.toR.weight, init.calculate_gain('relu'))
        
        
net1=Net1()

p=[]

#first using L1, and adadelta:
criterion = nn.L1Loss()
optimizer=torch.optim.Adadelta(net1.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
for epoch in range(20):
    running_loss=0.0
    for i in range(100):
        HR,LR=generator()
        optimizer.zero_grad()
        outputs=net1(LR)
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
            outputs=net1(LR_test).data.squeeze()
            PSNR+=psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    p.append(PSNR)
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
print('Finished Training phase1')

#next using L2, and adam:
criterion = nn.MSELoss()
optimizer=torch.optim.Adam(net1.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for epoch in range(20):
    running_loss=0.0
    for i in range(100):
        HR,LR=generator()
        optimizer.zero_grad()
        outputs=net1(LR)
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
            outputs=net1(LR_test).data.squeeze()
            PSNR+=psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    p.append(PSNR)
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
print('Finished Training phase1')


plt.plot(list(range(1,len(p)+1)), p)
plt.title('net1')
plt.xlabel('epoch')
plt.ylabel('PSNR')
plt.show
##saving and loading:
torch.save(net1.state_dict(), 'real_net1.pt')
#net1= Net1()
#net1.load_state_dict(torch.load('/home/lee/Desktop/python/sillyman/real_net1.pt'))
#net1.eval()
