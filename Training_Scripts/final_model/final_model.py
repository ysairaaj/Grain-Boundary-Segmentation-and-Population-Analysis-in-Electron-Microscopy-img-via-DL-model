import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import numpy as np
#from torchinfo import summary
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from network import UNet16


class Im_Seg_2(torch.nn.Module):
    def __init__(self ,device, loc = "./models/" , trainable = set()):
        super().__init__()
        #self.a = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        #self.b = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        #self.c = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.a = torch.nn.Parameter(torch.Tensor([0.6 , 0.2 ,0.2]) , requires_grad=True)
        # self.b = torch.nn.Parameter(torch.tensor(0.2) , requires_grad=True)
        # self.c = torch.nn.Parameter(torch.tensor(0.2) , requires_grad=True)
        #self.d = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.mod1 = UNet16() 
        self.mod2 = UNet16() 
        self.mod3 = UNet16() 
        self.mod1.to(device = device)
        self.mod2.to(device = device)
        self.mod3.to(device = device)
        print("Reading model 1")
        self.mod1.load_state_dict(torch.load(loc + "model1.pt", map_location=device))
        print("Reading model 2")
        self.mod2.load_state_dict(torch.load(loc + "model2.pt", map_location=device))
        print("Reading model 3")
        self.mod3.load_state_dict(torch.load(loc + "model3.pt", map_location=device))
        for k in self.mod1.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False
        for k in self.mod2.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False  
        for k in self.mod3.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False  

        #print(self.a.requires_grad , self.b.requires_grad , self.c.requires_grad)

        #self.a.requires_grad = True 
        #self.b.requires_grad = True 
        #self.c.requires_grad = True 
        #print(len(list(self.check_state(1))) ,len(list(self.check_state(2))) , len(list(self.check_state(3))) )
    def check_state(self , mod):
        dic = {1 : self.mod1 , 2 : self.mod2 , 3 : self.mod3 }
        return dic[mod].named_parameters()
    
    def forward(self , inp1 , inp2 ,inp3):
        #print("Yes , Yes , Yes")
        new = torch.nn.functional.softmax(self.a , dim = 0)
        out1 = self.mod1(inp1)
        out2 = self.mod2(inp2)
        out3 = self.mod3(inp3)
        #return float(20.0)*torch.sigmoid(out1 + out2 + out3)
        #print("Nice")
        return new[0]*out1 + new[1]*out2 + new[2]*out3
    

class Im_Seg_3(torch.nn.Module):
    def __init__(self ,device, loc = "./models/" , trainable = set()):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.c = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        #self.a = torch.nn.Parameter(torch.Tensor([0.6 , 0.2 ,0.2]) , requires_grad=True)
        # self.b = torch.nn.Parameter(torch.tensor(0.2) , requires_grad=True)
        # self.c = torch.nn.Parameter(torch.tensor(0.2) , requires_grad=True)
        #self.d = torch.nn.Parameter(torch.randn(()) , requires_grad=True)
        self.mod1 = UNet16() 
        self.mod2 = UNet16() 
        self.mod3 = UNet16() 
        self.mod1.to(device = device)
        self.mod2.to(device = device)
        self.mod3.to(device = device)
        print("Reading model 1")
        self.mod1.load_state_dict(torch.load(loc + "model1.pt", map_location=device))
        print("Reading model 2")
        self.mod2.load_state_dict(torch.load(loc + "model2.pt", map_location=device))
        print("Reading model 3")
        self.mod3.load_state_dict(torch.load(loc + "model3.pt", map_location=device))
        for k in self.mod1.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False
        for k in self.mod2.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False  
        for k in self.mod3.named_parameters():
            k[1].requires_grad = True if k[0].split(".")[0] in trainable else False  

        #print(self.a.requires_grad , self.b.requires_grad , self.c.requires_grad)

        #self.a.requires_grad = True 
        #self.b.requires_grad = True 
        #self.c.requires_grad = True 
        #print(len(list(self.check_state(1))) ,len(list(self.check_state(2))) , len(list(self.check_state(3))) )
    def check_state(self , mod):
        dic = {1 : self.mod1 , 2 : self.mod2 , 3 : self.mod3 }
        return dic[mod].named_parameters()
    
    def forward(self , inp1 , inp2 ,inp3):
        #print("Yes , Yes , Yes")
        #new = torch.nn.functional.softmax(self.a , dim = 0)
        out1 = self.mod1(inp1)
        out2 = self.mod2(inp2)
        out3 = self.mod3(inp3)
        #return float(20.0)*torch.sigmoid(out1 + out2 + out3)
        #print("Nice")
        return float(20.0)*torch.sigmoid(self.a*out1 + self.b*out2 + self.c*out3)
        #return new[0]*out1 + new[1]*out2 + new[2]*out3    



class Im_Seg_4(torch.nn.Module):
    def __init__(self ,device, loc = "./models/" , trainable = set() , dim = 256):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(3 , dim*dim) , requires_grad=True)
        self.mod1 = UNet16() 
        self.mod2 = UNet16() 
        self.mod3 = UNet16() 
        self.mod1.to(device = device)
        self.mod2.to(device = device)
        self.mod3.to(device = device)
        print("Reading model 1")
        self.mod1.load_state_dict(torch.load(loc + "model1.pt", map_location=device))
        print("Reading model 2")
        self.mod2.load_state_dict(torch.load(loc + "model2.pt", map_location=device))
        print("Reading model 3")
        self.mod3.load_state_dict(torch.load(loc + "model3.pt", map_location=device))
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
        new = torch.nn.functional.softmax(self.a , dim = 0)
        out1 = self.mod1(inp1)
        B , C ,H ,W = out1.shape
        out2 = self.mod2(inp2)
        out3 = self.mod3(inp3)
        out1_flat = torch.flatten(out1 , start_dim=2)
        out2_flat = torch.flatten(out2 , start_dim=2)
        out3_flat = torch.flatten(out3 , start_dim=2)
        out1_dot = torch.mul(out1_flat , new[0])
        out2_dot = torch.mul(out2_flat , new[1])
        out3_dot = torch.mul(out3_flat , new[2])
        out_dot = out1_dot + out2_dot + out3_dot
        return torch.reshape(out_dot ,(B,C,H,W) )