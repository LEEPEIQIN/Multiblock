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

cwd=os.getcwd()

LR_URL='2_LR'
HR_URL='2_HR'



LR_set=torchvision.datasets.ImageFolder(root=LR_URL)

os.mkdir('2_LR_patch')
os.chdir('2_LR_patch')
os.mkdir('2')
os.chdir('2')
wd=os.getcwd()

os.chdir(cwd)

for i in range(len(LR_set)):
    img_1=Image.open(LR_set.imgs[i][0])
    img_2=torchvision.transforms.functional.rotate(img_1,90)
    img_3=torchvision.transforms.functional.rotate(img_1,180)
    img_4=torchvision.transforms.functional.rotate(img_1,270)
    img_5=img_1.transpose(Image.FLIP_LEFT_RIGHT)
    img_6=img_2.transpose(Image.FLIP_LEFT_RIGHT)
    img_7=img_3.transpose(Image.FLIP_LEFT_RIGHT)
    img_8=img_4.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    img_1=torchvision.transforms.functional.to_tensor(img_1).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_1,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img1_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_2=torchvision.transforms.functional.to_tensor(img_2).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_2,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img2_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_3=torchvision.transforms.functional.to_tensor(img_3).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_3,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img3_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_4=torchvision.transforms.functional.to_tensor(img_4).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_4,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img4_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_5=torchvision.transforms.functional.to_tensor(img_5).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_5,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img5_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_6=torchvision.transforms.functional.to_tensor(img_6).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_6,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img6"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_7=torchvision.transforms.functional.to_tensor(img_7).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_7,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img7_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_8=torchvision.transforms.functional.to_tensor(img_8).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_8,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img8_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    print("finish picture [%d]" % i)
    
    
    
    
    
    
    
    
    
    
    

HR_set=torchvision.datasets.ImageFolder(root=HR_URL)

os.mkdir('2_HR_patch')
os.chdir('2_HR_patch')
os.mkdir('2')
os.chdir('2')
wd=os.getcwd()

os.chdir(cwd)

for i in range(len(HR_set)):
    img_1=Image.open(HR_set.imgs[i][0])
    img_2=torchvision.transforms.functional.rotate(img_1,90)
    img_3=torchvision.transforms.functional.rotate(img_1,180)
    img_4=torchvision.transforms.functional.rotate(img_1,270)
    img_5=img_1.transpose(Image.FLIP_LEFT_RIGHT)
    img_6=img_2.transpose(Image.FLIP_LEFT_RIGHT)
    img_7=img_3.transpose(Image.FLIP_LEFT_RIGHT)
    img_8=img_4.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    img_1=torchvision.transforms.functional.to_tensor(img_1).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_1,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img1_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_2=torchvision.transforms.functional.to_tensor(img_2).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_2,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img2_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_3=torchvision.transforms.functional.to_tensor(img_3).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_3,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img3_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_4=torchvision.transforms.functional.to_tensor(img_4).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_4,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img4_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_5=torchvision.transforms.functional.to_tensor(img_5).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_5,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img5_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_6=torchvision.transforms.functional.to_tensor(img_6).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_6,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img6"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_7=torchvision.transforms.functional.to_tensor(img_7).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_7,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img7_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    img_8=torchvision.transforms.functional.to_tensor(img_8).unsqueeze(0)
    temp=torch.nn.functional.unfold(img_8,(121,121),stride=(20,20))
    os.chdir(wd)
    for j in range(temp.size()[2]):
        patch=temp[:,:,j].reshape([1,3,121,121])
        patch=patch.squeeze()
        patch=transforms.ToPILImage()(patch)
        patch.save("2_LR_img8_"+str(i+1)+"_"+str(j+1)+".png","PNG")
    os.chdir(cwd)
    
    print("finish picture [%d]" % i)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    