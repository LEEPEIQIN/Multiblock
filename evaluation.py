#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import common

#test loader:
URL='real_net1_plus.pt'
test_LR = torchvision.datasets.ImageFolder(root='2_LR_test', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)

net_plus=common.Net_plus()
net_plus.load_state_dict(torch.load(URL))
net_plus.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_plus.to(device)

n_test=15
PSNR=0.0
temp_HR_2=iter(HR_2_test)
temp_LR_2=iter(LR_2_test)
with torch.no_grad():
    for t in range(n_test):
        HR_test=temp_HR_2.next()[0].squeeze()
        _,x,y=HR_test.size()
        LR_test=temp_LR_2.next()[0]
        if x>1500:
            x=1500
        if y>1500:
            y=1500
        HR_test=HR_test[:,0:x,0:y].to(device)
        LR_test=LR_test[:,:,0:x,0:y].to(device)
        outputs=net_plus(LR_test).data.squeeze()
        PSNR+=common.psnr(outputs,HR_test)
        del HR_test,LR_test,outputs
PSNR=PSNR/n_test
print(PSNR)

#test loader:
URL='real_net2_plus.pt'
test_LR = torchvision.datasets.ImageFolder(root='2_LR_test_afternet1', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)

net_plus=common.Net_plus()
net_plus.load_state_dict(torch.load(URL))
net_plus.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_plus.to(device)

n_test=15
PSNR=0.0
temp_HR_2=iter(HR_2_test)
temp_LR_2=iter(LR_2_test)
with torch.no_grad():
    for t in range(n_test):
        HR_test=temp_HR_2.next()[0].squeeze()
        _,x,y=HR_test.size()
        LR_test=temp_LR_2.next()[0]
        if x>1500:
            x=1500
        if y>1500:
            y=1500
        HR_test=HR_test[:,0:x,0:y].to(device)
        LR_test=LR_test[:,:,0:x,0:y].to(device)
        outputs=net_plus(LR_test).data.squeeze()
        PSNR+=common.psnr(outputs,HR_test)
        del HR_test,LR_test,outputs
PSNR=PSNR/n_test
print(PSNR)

#test loader:
URL='real_net3_plus.pt'
test_LR = torchvision.datasets.ImageFolder(root='2_LR_test_afternet2', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)

net_plus=common.Net_plus()
net_plus.load_state_dict(torch.load(URL))
net_plus.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_plus.to(device)

n_test=15
PSNR=0.0
temp_HR_2=iter(HR_2_test)
temp_LR_2=iter(LR_2_test)
with torch.no_grad():
    for t in range(n_test):
        HR_test=temp_HR_2.next()[0].squeeze()
        _,x,y=HR_test.size()
        LR_test=temp_LR_2.next()[0]
        if x>1500:
            x=1500
        if y>1500:
            y=1500
        HR_test=HR_test[:,0:x,0:y].to(device)
        LR_test=LR_test[:,:,0:x,0:y].to(device)
        outputs=net_plus(LR_test).data.squeeze()
        PSNR+=common.psnr(outputs,HR_test)
        del HR_test,LR_test,outputs
PSNR=PSNR/n_test
print(PSNR)