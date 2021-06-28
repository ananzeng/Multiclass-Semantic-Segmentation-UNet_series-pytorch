import torch
import os
import cv2 
import numpy as np 
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
from Model_Zoo import FCN_Model, ResUnet, Unet_Plus

#set up model
fcn8s = FCN_Model.FNC_8S(input_channel = 3,output_channel = 5)
fcn32s = FCN_Model.FNC_32S(input_channel = 3,output_channel = 5)
unet = Unet_Plus.UNet(input_channel = 3,output_channel = 5)
unet_plus = Unet_Plus.NestedUNet(input_channel = 3,output_channel = 5)
resunet = ResUnet.ResUNet(input_channel = 3,output_channel = 5)
deepresunet = ResUnet.DeepResUNet(input_channel = 3,output_channel = 5)

#set up hyper
epoch = 301
image_size = (256,256)
lr = 1e-2
weight_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set up dataset 
from  Dataset import CustomDataset
train_dataset = CustomDataset("dataset/data_shape/train_image", "dataset/data_shape/train_annot_mask", image_size = image_size)
vaild_dataset = CustomDataset("dataset/data_shape/vaild_image", "dataset/data_shape/vaild_annot_mask", image_size = image_size)
test_dataset = CustomDataset("dataset/data_shape/test_image", "dataset/data_shape/test_annot_mask", image_size = image_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=1, shuffle=False)
test1_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

#set up random seed
seed = 69
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train():
    #fcn8s, fcn32s, unet, unet_plus, resunet, deepresunet
    cnn = fcn8s.to(device)
    print(cnn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=lr,  weight_decay=weight_decay)

    prev_time = datetime.now()
    array_train_loss = []
    array_vaild_loss = []
    for epo in range(epoch):
        train_loss = 0
        cnn.train()

        for index, (img, mask, mask_path) in enumerate(train_loader):
            img = img.to(device)                        #把img送到gpu
            mask = mask.to(device)                      #把mask送到gpu
            mask = mask.long()
            optimizer.zero_grad()                       #梯度歸零
            output = cnn(img)                           #計算輸出
            loss = criterion(output, mask)              #算loss
            loss.backward()                             #反向傳播
            iter_loss = loss.item()                     #取出loss
            train_loss += iter_loss
            optimizer.step()                            #更新權重
            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_loader), iter_loss))


        test_loss = 0
        cnn.eval()
        with torch.no_grad():
            for index, (img, mask, mask_path) in enumerate(vaild_loader):
                img = img.to(device)
                mask = mask.to(device)
                mask = mask.long()
                optimizer.zero_grad()
                output = cnn(img)
                loss = criterion(output, mask)
                iter_loss = loss.item()
                test_loss += iter_loss
            
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch vaild loss = %f, %s' %(train_loss/len(train_loader), test_loss/len(vaild_loader), time_str))
        array_train_loss.append(train_loss/len(train_loader))
        array_vaild_loss.append(test_loss/len(vaild_loader))
    
        if np.mod(epo, 5) == 0:
            torch.save(cnn, 'checkpoints/model_{}.pt'.format(epo))
            print('saveing checkpoints/model_{}.pt'.format(epo))

    plt.figure(figsize=(6,4),dpi=100,linewidth = 2)
    plt.plot(np.arange(epoch),array_train_loss,'o-',color = 'b', label="train_loss")
    plt.plot(np.arange(epoch),array_vaild_loss,'o-',color = 'orange', label="vaild_loss")
    plt.title("Model performance")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc = "best", fontsize=10)
    plt.savefig('checkpoints/result.jpg')
if __name__ == '__main__':
    train()