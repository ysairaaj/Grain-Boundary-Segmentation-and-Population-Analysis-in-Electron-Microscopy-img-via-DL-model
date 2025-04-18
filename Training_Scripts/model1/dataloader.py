import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import PIL 
import PIL.Image
import numpy as np
import os

class Data_from_disk(torch.utils.data.Dataset) :
    def __init__(self ,typ = "train", dir_ = "./data/"): #tag indicates the type of input image to be trained on 
        super().__init__()
        self.dir_ = dir_ 
        self.typ = typ
        self.inp_dir = tuple(sorted(os.listdir(dir_+ "_inp_/" + typ + "/")))
        self.out_dir = tuple(sorted(os.listdir(dir_ + "_out_/" + typ + "/")))
        self.channel_first = True
        assert self.inp_dir == self.out_dir , "Images are not arranged right ."
    def __getitem__(self , idx):
        #print(self.inp_dir[idx] , self.out_dir[idx])
        #assert self.inp_dir[idx] == self.out_dir[idx] , f"names dont match . given names are {self.inp_dir[idx]} and {self.out_dir[idx]}"
        #assert self.inp_dir[idx].split("_")[2] == str(self.tag) , f"tags are different , required : {self.tag} , found : {self.inp_dir[idx].split('_')[2]}"
        x1 = np.array(PIL.Image.open(self.dir_ + "_inp_/" + self.typ + '/' + self.inp_dir[idx]))
        x2 = np.array(PIL.Image.open(self.dir_ + "_out_/" + self.typ + "/" + self.out_dir[idx]))[:,:,0:1]
        if self.channel_first :
            x1 = np.transpose(x1 ,(2, 0, 1))
            x2 = np.transpose(x2 ,(2, 0, 1))
        x1 = (x1/255).astype(np.float32)
        x2 = (x2/255).astype(np.float32)
        meanR, meanG, meanB = .485,.456,.406
        stdR, stdG, stdB = .229, .224, .225 
        norm_= T.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
        #if return_names :
        #return (torch.from_numpy(x1) , torch.from_numpy(x2)) , (self.inp_dir[idx] ,self.out_dir[idx] )
        return (norm_(torch.from_numpy(x1)) , torch.from_numpy(x2))
    
    def __len__(self):
        return (len(self.inp_dir)) 
    
    def channel_last(self):
        self.channel_first = False 

    def channel_first(self):
        self.channel_first = True  
