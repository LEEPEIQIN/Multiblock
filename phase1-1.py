#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import common

#test loader:
test_LR = torchvision.datasets.ImageFolder(root='2_LR_test', transform=transforms.ToTensor())
LR_2_test = torch.utils.data.DataLoader(test_LR, batch_size=1, shuffle=False, num_workers=0)
test_HR = torchvision.datasets.ImageFolder(root='2_HR_test', transform=transforms.ToTensor())
HR_2_test = torch.utils.data.DataLoader(test_HR, batch_size=1, shuffle=False, num_workers=0)

net1=common.Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net1.to(device)


criterion = nn.L1Loss()
optimizer=torch.optim.Adam(net1.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for epoch in range(4):
    running_loss=0.0
    for i in range(500):
        HR,LR=common.generator("2_HR","2_LR",16,48)
        HR=HR.to(device)
        LR=LR.to(device)
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
            outputs=net1(LR_test).data.squeeze()
            PSNR+=common.psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
    torch.save(net1.state_dict(), 'real_net1.pt')
print('Finished Training net')



del net1
net1_plus=common.inherit('real_net1.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net1_plus.to(device)


criterion = nn.L1Loss()
optimizer=torch.optim.Adam(net1_plus.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for epoch in range(6):
    running_loss=0.0
    for i in range(500):
        HR,LR=common.generator("2_HR","2_LR",16,48)
        HR=HR.to(device)
        LR=LR.to(device)
        optimizer.zero_grad()
        outputs=net1_plus(LR)
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
            outputs=net1_plus(LR_test).data.squeeze()
            PSNR+=common.psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
    torch.save(net1_plus.state_dict(), 'real_net1_plus.pt')
print('Finished Training 1')

criterion = nn.L1Loss()
optimizer=torch.optim.Adam(net1_plus.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for epoch in range(20):
    running_loss=0.0
    for i in range(500):
        HR,LR=common.generator("2_HR","2_LR",16,48)
        HR=HR.to(device)
        LR=LR.to(device)
        optimizer.zero_grad()
        outputs=net1_plus(LR)
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
            outputs=net1_plus(LR_test).data.squeeze()
            PSNR+=common.psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    print(PSNR)
    del PSNR,n_test,temp_HR_2,temp_LR_2
    torch.save(net1_plus.state_dict(), 'real_net1_plus.pt')
print('Finished Training 2')


