import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torchinfo import summary
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from Convs_Unet import UNet


class Im_Seg(torch.nn.Module):
    def __init__(self ,device, loc = "../models/" , trainable = set()):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.c = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.d = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.mod1 = UNet(n_channels = 3 , n_classes = 1) 
        self.mod2 = UNet(n_channels = 3 , n_classes = 1) 
        self.mod3 = UNet(n_channels = 3 , n_classes = 1) 
        self.mod1.to(device = device)
        self.mod2.to(device = device)
        self.mod3.to(device = device)
        self.mod1.load_state_dict(torch.load(loc + "model1.pth", map_location=device))
        self.mod2.load_state_dict(torch.load(loc + "model2.pth", map_location=device))
        self.mod3.load_state_dict(torch.load(loc + "model3.pth", map_location=device))
        for k in self.mod1.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False
        for k in self.mod2.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False  
        for k in self.mod3.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False  
            
    def check_state(self , mod):
        dic = {1 : self.mod1 , 2 : self.mod2 , 3 : self.mod3 }
        return dic[mod].named_parameters()
    
    def forward(self , inp1 , inp2 ,inp3):
        out1 = torch.sigmoid(self.mod1(inp1))
        out2 = torch.sigmoid(self.mod2(inp2))
        out3 = torch.sigmoid(self.mod3(inp3))
        return torch.sigmoid(self.a*out1 + self.b*out2 + self.c*out3 + self.d)