import torch.utils.data as data
from torchvision import transforms

import torch
import cv2
import json
import numpy as np
import os
from PIL import Image


class CustomDataset(data.Dataset):
    def __init__(self, image_paths, annot_paths, image_size):   # initial logic happens like transform
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.image_size = image_size
        self.img = list(sorted(os.listdir(image_paths)))
        self.mask = list(sorted(os.listdir(annot_paths)))  
        self.transforms = transforms.ToTensor()
        self.mapping = {
            0: 0,  
            100: 1,
            250: 2,
            200: 3,
            255: 4
        }
    
    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask

    def __getitem__(self, index):
        img_path = os.path.join(self.image_paths, self.img[index])
        mask_path = os.path.join(self.annot_paths, self.mask[index])
        image = Image.open(img_path)
        #print(img_path)
        image = image.resize(self.image_size,Image.NEAREST)
        mask = Image.open(mask_path)
        mask = mask.resize(self.image_size,Image.NEAREST)
        t_image = self.transforms(image)
        mask = torch.from_numpy(np.array(mask))
        mask = self.mask_to_class(mask)
    
        return t_image, mask, self.mask[index]
        
    def __len__(self):  # return count of sample we have
        return len(self.img)

if __name__ == '__main__':
    for index,(data, target) in enumerate(train_loader):
        print("第",index+1,"個data")
        print("Target的shape",target.shape)  
        print(np.unique(target[0]))
        break