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
#torch.load with map_location=torch.device('cpu')
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

mapping = {
    0: 0,  
    1: 100,
    2: 250,
    3: 200,
    4: 255
}
def mask_to_class(mask):
    for k in mapping:
        mask[mask==k] = mapping[k]
    return mask

def test():
    average_f1 = 0
    average_recall = 0
    average_precision = 0
    print("資料總數：", len(test_loader))
    #model_name = os.path.join(os.getcwd(),'model/unet_self_model_60_shape.pt') #問學長model_load就夠了嗎?
    model_name = os.path.join(os.getcwd(),'checkpoints/unet_plus.pt') #問學長model_load就夠了嗎?
    cnn = torch.load(model_name,map_location ='cpu')
    optimizer = torch.optim.SGD(cnn.parameters(), lr=1e-2, weight_decay=1e-4)
    cnn.eval()
    with torch.no_grad():
        for index, (img, mask, mask_path) in enumerate(test_loader):#問學長demo的方法
            print(str(index+1) + "/" + str(len(test_loader)), end="  ")
            print("mask_path:", str(mask_path)[2:-3])
            img = img.to(device)
            optimizer.zero_grad()
            output = cnn(img)   
            pred = torch.argmax(output, 1)
            pred = pred.cpu().numpy()
            pic = mask_to_class(pred[0])
            #pic = cv2.resize(pred[0]*50, (256, 256), interpolation=cv2.INTER_NEAREST)
            Predict_path = os.path.join("dataset", "result", str(mask_path)[2:-3]+"_predict.png")
            print(Predict_path)
            cv2.imwrite(Predict_path,pic)

if __name__ == '__main__':
    test()