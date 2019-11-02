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

net_combine=common.inherit_combined('real_net1_plus.pt','real_net2_plus.pt','real_net3_plus.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_combine.to(device)


best=0.0
criterion = nn.L1Loss()
optimizer=torch.optim.Adam(net_combine.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for epoch in range(10):
    #running_loss=0.0
    for i in range(500):
        HR,LR=common.generator("2_HR","2_LR",8,120)
        HR=HR.to(device)
        LR=LR.to(device)
        optimizer.zero_grad()
        outputs=net_combine(LR)
        Loss=criterion(outputs,HR)
        Loss.backward()
        optimizer.step()
        #running_loss+=Loss.item()
        
        #print('[%d, %5d] loss: %.3f' %
        #     (epoch + 1, i + 1, running_loss))
        #running_loss = 0.0
        del HR,LR,outputs
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
            HR_test=HR_test[:,0:(x-1),0:(y-1)].to(device)
            LR_test=LR_test[:,:,0:(x-1),0:(y-1)].to(device)
            outputs=net_combine(LR_test).data.squeeze()
            PSNR+=common.psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    print(PSNR)
    if PSNR>best:
        best=PSNR
    print('best is %.4f' % best)
    del PSNR,n_test,temp_HR_2,temp_LR_2
    torch.save(net_combine.state_dict(), 'net_combine1_test.pt')
print('Finished Training 1')



criterion = nn.L1Loss()
optimizer=torch.optim.Adam(net_combine.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for epoch in range(10):
    #running_loss=0.0
    for i in range(500):
        HR,LR=common.generator("2_HR","2_LR",8,120)
        HR=HR.to(device)
        LR=LR.to(device)
        optimizer.zero_grad()
        outputs=net_combine(LR)
        Loss=criterion(outputs,HR)
        Loss.backward()
        optimizer.step()
        #running_loss+=Loss.item()
        
        #print('[%d, %5d] loss: %.3f' %
        #     (epoch + 1, i + 1, running_loss))
        #running_loss = 0.0
        del HR,LR,outputs
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
            HR_test=HR_test[:,0:(x-1),0:(y-1)].to(device)
            LR_test=LR_test[:,:,0:(x-1),0:(y-1)].to(device)
            outputs=net_combine(LR_test).data.squeeze()
            PSNR+=common.psnr(outputs,HR_test)
            del HR_test,LR_test,outputs
    PSNR=PSNR/n_test
    print(PSNR)
    if PSNR>best:
        best=PSNR
    print('best is %.4f' % best)
    del PSNR,n_test,temp_HR_2,temp_LR_2
    torch.save(net_combine.state_dict(), 'net_combine1_test.pt')
print('Finished Training 2')