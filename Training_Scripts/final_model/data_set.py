"""

This script contains the codes to segment cracks.
The codes are based on "TOPO-Loss for continuity-preserving crack detection using deep learning" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.conbuildmat.2022.128264

This script specifically support deep learning codes development.
These are based on codes published in:
"Avendi, M., 2020. PyTorch Computer Vision Cookbook:
Over 70 Recipes to Master the Art of Computer Vision with Deep Learning and PyTorch 1. x. Packt Publishing Limited."

Slightly changes are introduced to addapt to general pipeline

@author: pantoja
"""

# Import Modules
import os
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy

random.seed(100)

#Define the open_dataset class:
class open_dataset(Dataset):
    def __init__(self, path2data_i , path2data_i2 , path2data_i3, path2data_m=None, transform=None):      

        imgsList=[pp for pp in os.listdir(path2data_i)]
        imgsList.sort()
        imgsList = tuple(imgsList)
        imgsList2=[pp for pp in os.listdir(path2data_i2)]
        imgsList2.sort()
        imgsList2 = tuple(imgsList2)
        imgsList3=[pp for pp in os.listdir(path2data_i3)]
        imgsList3.sort()
        imgsList3 = tuple(imgsList3)
        if transform!="test":
            anntsList=[pp for pp in os.listdir(path2data_m)]
            anntsList.sort()
            anntsList = tuple(anntsList)
        if transform!="test":
            assert imgsList == anntsList and imgsList2 == anntsList and imgsList3 == anntsList, "Train sets inputs and outputs dont coincide ."

        self.path2imgs = [os.path.join(path2data_i, fn) for fn in imgsList] 
        self.path2imgs2 = [os.path.join(path2data_i2, fn) for fn in imgsList2] 
        self.path2imgs3 = [os.path.join(path2data_i3, fn) for fn in imgsList3] 

        if transform!="test":
            self.path2annts= [os.path.join(path2data_m, fn) for fn in anntsList]

        self.transform = transform
    
    def __len__(self):
        return len(self.path2imgs)
      
    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        path2img2 = self.path2imgs2[idx]
        path2img3 = self.path2imgs3[idx]
        img = Image.open(path2img).convert('RGB')
        img2 = Image.open(path2img2).convert('RGB')
        img3 = Image.open(path2img3).convert('RGB')
        if self.transform!="test":
            path2annt = self.path2annts[idx]
            mask = Image.open(path2annt)
                
        if self.transform=='train':
                # if random.random()<.5:
                #     img = TF.hflip(img)
                #     mask = TF.hflip(mask)
                # if random.random()<.5:
                #     img = TF.vflip(img)
                #     mask = TF.vflip(mask)
                if random.random()<.5:
                    img = TF.adjust_brightness(img,brightness_factor=.5)
                    img2 = TF.adjust_brightness(img2,brightness_factor=.5)
                    img3 = TF.adjust_brightness(img3,brightness_factor=.5)
                if random.random()<.5:
                    img = TF.adjust_contrast(img,contrast_factor=.4)
                    img2 = TF.adjust_contrast(img2,contrast_factor=.4)
                    img3 = TF.adjust_contrast(img3,contrast_factor=.4)
                if random.random()<.5:
                    img = TF.adjust_gamma(img,gamma=1.4)
                    img2 = TF.adjust_gamma(img2,gamma=1.4)
                    img3 = TF.adjust_gamma(img3,gamma=1.4)
                if random.random()<.5:
                    trans = T.Grayscale(num_output_channels=3)
                    img = trans(img)
                    img2 = trans(img2)
                    img3 = trans(img3)
                if random.random()<.0:
                    trans = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
                    img = trans(img)
                    img2 = trans(img2)
                    img3 = trans(img3)
        
        if self.transform!='test':
            im_size = 256
            trans = T.Resize((im_size,im_size))
            img = trans(img)
            img2 = trans(img2)
            img3 = trans(img3)

        if self.transform!="test": mask = trans(mask)
        trans = T.ToTensor()
        img = trans(img)
        img2 = trans(img2)
        img3 = trans(img3)
        if self.transform!="test":
            mask = np.array(mask) #to array
            mask = (mask > 200) #background zero. 1-mask
            mask = mask[:,:,0]
            mask=scipy.ndimage.distance_transform_edt(mask) #creating distance map
            mask[mask>20] = 20 #Cliping the distance map
            mask = trans(mask)
        
        #VGG16 mean and std 
        meanR, meanG, meanB = .485,.456,.406
        stdR, stdG, stdB = .229, .224, .225 
        norm_= T.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
        img = norm_(img)
        img2 = norm_(img2)
        img3 = norm_(img3)
        
        if self.transform!='test':
            return (img , img2 , img3), mask
        else:
            return (img , img2 , img3)