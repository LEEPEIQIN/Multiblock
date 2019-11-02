#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import os

import common

net_plus=common.Net_plus()
net_plus.load_state_dict(torch.load('real_net1_plus.pt'))
net_plus.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_plus.to(device)

cwd = os.getcwd()

#####################################################################3
LR_URL='2_LR'

LR_set=torchvision.datasets.ImageFolder(root=LR_URL,transform=transforms.ToTensor())
LR_loader=torch.utils.data.DataLoader(LR_set,batch_size=1,shuffle=False,num_workers=0)
LR_loader=iter(LR_loader)

os.mkdir('2_LR_afternet1')
os.chdir('2_LR_afternet1')
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
        LR_1=net_plus(LR_tl).data
        LR_2=net_plus(LR_tr).data
        LR_3=net_plus(LR_bl).data
        LR_4=net_plus(LR_br).data
        temp1=torch.cat((LR_1,LR_2),3)
        temp2=torch.cat((LR_3,LR_4),3)
        temp=torch.cat((temp1,temp2),2).squeeze()
        LR_after=temp.cpu()
        LR_after = transforms.ToPILImage()(LR_after)
        os.chdir(wd_1)
        if i<9:
            LR_after.save("2_LR_afternet1_00"+str(i+1)+".png","PNG")
        else:
            LR_after.save("2_LR_afternet1_0"+str(i+1)+".png","PNG")
        os.chdir(cwd)
        print('done')
        
##
        
LR_URL='2_LR_test'

LR_set=torchvision.datasets.ImageFolder(root=LR_URL,transform=transforms.ToTensor())
LR_loader=torch.utils.data.DataLoader(LR_set,batch_size=1,shuffle=False,num_workers=0)
LR_loader=iter(LR_loader)

os.mkdir('2_LR_test_afternet1')
os.chdir('2_LR_test_afternet1')
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
       LR_1=net_plus(LR_tl).data
       LR_2=net_plus(LR_tr).data
       LR_3=net_plus(LR_bl).data
       LR_4=net_plus(LR_br).data
       temp1=torch.cat((LR_1,LR_2),3)
       temp2=torch.cat((LR_3,LR_4),3)
       temp=torch.cat((temp1,temp2),2).squeeze()
       LR_after=temp.cpu()
       LR_after = transforms.ToPILImage()(LR_after)
       os.chdir(wd_2)
       if i<9:
            LR_after.save("2_LR_test_afternet1_00"+str(i+1)+".png","PNG")
       else:
            LR_after.save("2_LR_test_afternet1_0"+str(i+1)+".png","PNG")
       os.chdir(cwd)
       print('done')