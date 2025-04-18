import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torchinfo import summary
from torch.utils.data.dataloader import DataLoader
#from torch.data import random_split
from torchvision import transforms
from Convs_Unet import UNet 
from torch.utils.data import random_split 
from tqdm import trange, tqdm

tag_name = 0

model = UNet(n_channels = 3 , n_classes = 1)
print("Loading Unet model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")
model.to(device=device)
model.load_state_dict(torch.load("../Tiles_512_Sigmod6CT_ep150.pth", map_location=device))
print("model loaded")

trainable = {"inc" : False , "down1" : False}
for k in model.named_parameters():
    k[1].requires_grad = False if k[0].split(".")[0] in trainable else True
    
class Data_from_disk(torch.utils.data.Dataset) :
    def __init__(self , tag = 0 , dir_ = "../Data/Full_Aug/"): #tag indicates the type of input image to be trained on 
        super().__init__()
        self.tag = tag 
        self.dir_ = dir_
        self.inp_dir = sorted([x for x in os.listdir(dir_ + "Im_inp/") if x.split("_")[2] == str(tag)])
        self.out_dir = sorted(os.listdir(dir_ + "Im_out/"))
    def __getitem__(self , idx):
        #print(self.inp_dir[idx] , self.out_dir[idx])
        #assert self.inp_dir[idx] == self.out_dir[idx] , f"names dont match . given names are {self.inp_dir[idx]} and {self.out_dir[idx]}"
        #assert self.inp_dir[idx].split("_")[2] == str(self.tag) , f"tags are different , required : {self.tag} , found : {self.inp_dir[idx].split('_')[2]}"
        x1 = np.array(Image.open(self.dir_ + "Im_inp/" + self.inp_dir[idx]))
        x2 = np.array(Image.open(self.dir_ + "Im_out/" + self.out_dir[idx]))[:,:,0:1]
        x1 = np.transpose(x1 ,(2, 0, 1))
        x2 = np.transpose(x2 ,(2, 0, 1))
        x1 = x1/255
        x2 = x2/255
        #if return_names :
            #return (torch.from_numpy(x1) , torch.from_numpy(x2)) , (self.inp_dir[idx] ,self.out_dir[idx] )
        return (torch.from_numpy(x1) , torch.from_numpy(x2))
    
    def __len__(self):
        return (len(self.inp_dir)) 

#Plan is to train three different models on the training data of three tags and use their ensemble

g1 = torch.Generator().manual_seed(42)
train_ds , val_ds , test_ds = random_split(Data_from_disk(tag = tag_name) , [0.7 , 0.2 ,0.1] , generator = g1)
print(f"train_dataset = {len(train_ds)} , val_dataset = {len(val_ds)} , test_dataset = {len(test_ds)}")
batch = 3
train_loader = DataLoader(train_ds , batch , shuffle = True)
val_loader = DataLoader(val_ds , batch)
test_loader = DataLoader(test_ds , batch)

for x , y in train_loader:
    print(x.shape , y.shape)
    break

EPOCHS = 100
lr = 0.01

def evaluate(model , val_loader ,loss_func ):
    curr_val_loss = 0
    with torch.no_grad() :
        for i ,mm in enumerate(val_loader):
            x = mm[0]
            y = mm[1]
            x = x.to(device , dtype = torch.float32)
            y = y.to(device , dtype = torch.float32)
            loss = loss_func(torch.sigmoid(model(x)), y)
            curr_val_loss = (curr_val_loss*(i) + loss)/(i+1)
    return curr_val_loss 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss*(self.min_delta + 1)):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False   


#from collections import deque
early_stopper = EarlyStopper(patience = 3 , min_delta = 0.05)
def fit(model , epochs , lr , train_loader , val_loader , opt_func , loss_func ):
    #Q = deque()
    history = []
    optimizer = opt_func(model.parameters() , lr)
    cur_val_loss = 0
    scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.95 , verbose = True)  
    for epoch in tqdm(range(epochs) , total = epochs , desc="num_epochs") :
        iterator = tqdm(enumerate(train_loader) , total = len(train_loader) ,desc="num_batches") 
        for i ,mm in iterator:
            x = mm[0]
            y = mm[1]
            x = x.to(device , non_blocking = True , dtype = torch.float32)
            y = y.to(device , non_blocking = True , dtype = torch.float32)
            #scheduler.print_lr()
            #print(x.device , y.device)
            optimizer.zero_grad()
            #print("zero_grad")
            loss = loss_func(torch.sigmoid(model(x)), y)
            #print("calc_loss")
            iterator.set_postfix(train_loss = loss.item())
            loss.backward()
            optimizer.step()
            
            #torch.cuda.empty_cache()
        scheduler.step()    
        val_err = evaluate(model , val_loader , loss_func)
        curr_train_loss = evaluate(model , train_loader , loss_func)
        print(f"train_loss : {curr_train_loss}  , val_loss : {val_err}")
        history.append((curr_train_loss , val_err ))
        #if len(Q) == 3 :
        #    Q.popleft()
        #Q.append(model.parameters())
        if early_stopper.early_stop(val_err) :
            #model.parameters() = Q[0]
            model_name = f"model{tag_name + 1}.pth"
            torch.save(model.state_dict(), model_name)  
            print(f"Saved model to model{tag_name + 1}.pth")
            history = np.array([(x111[0].item() , x111[1].item()) for x111 in history])
            his_name = f"history{tag_name + 1}.csv"
            np.savetxt(his_name, history, delimiter=',')
            break
          
    model_name = f"model{tag_name + 1}.pth"
    torch.save(model.state_dict(), model_name)  
    print(f"Saved model to model{tag_name + 1}.pth")
    history = np.array([(x111[0].item() , x111[1].item()) for x111 in history])
    his_name = f"history{tag_name + 1}.csv"
    np.savetxt(his_name, history, delimiter=',')
    
    return history        


fit_dic1 = {"model" : model ,
           "epochs" : EPOCHS ,
           "lr" : lr ,
           "train_loader" : train_loader ,
           "val_loader" : val_loader ,
           "opt_func" : torch.optim.Adam  ,
           "loss_func" : torch.nn.functional.binary_cross_entropy}

history = fit(**fit_dic1)

print("Going for round 2")
trainable = {}
for k in model.named_parameters():
    k[1].requires_grad = False if k[0].split(".")[0] in trainable else True

fit_dic1["epochs"] = 30
fit_dic1["lr"] = fit_dic1[lr]/2

history = fit(**fit_dic1)

print("Model trained !!! ")
print("Evaluating on test set ...")
ans = evaluate(model , test_loader ,torch.nn.functional.binary_cross_entropy )

print(f"Test set score : {ans}")

print("exiting ...")
