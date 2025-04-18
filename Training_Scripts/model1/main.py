import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
from network import UNet16
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import optim
#from loss_opt import *
from training_utils import *
from torch import nn
from newloss import MALISLoss_window_pos
import argparse
from train_data_generator import *
from dataloader import *
from training_utils import *
from data_set import open_dataset
from Convs_Unet import UNet

#Setting input hyperparameters with argparse
parser = argparse.ArgumentParser(description='Grain_segementation_trainer')
parser.add_argument('--lr', action='store', default=0.0001, help='Learning rate')
parser.add_argument('--n_epoch', action='store', default = 15, help='Number of epoch')
parser.add_argument('--malis_neg', action='store', default = 1.0, help='Negative Malis parameter')
parser.add_argument('--malis_pos', action='store', default = 1.0, help='Positive Malis parameter')
parser.add_argument('--tag', action='store', default = 0, help='tag')
args = parser.parse_args()

#Hyperparameters
lr = float(args.lr)
n_epoch = int(args.n_epoch)
malis_neg = float(args.malis_neg)
malis_pos = float(args.malis_pos)
tag = int(args.tag)

# To have reproducible results the random seed is set to 42.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


#Paths to images and their masks
#train
path2train_i="./data/_inp_/train"
path2train_m="./data/_out_/train"
#validation
path2valid_i="./data/_inp_/val"
path2valid_m="./data/_out_/val" 
#test
path2test_i="./data/_inp_/test"
path2test_m="./data/_out_/test" 

##DATA LOADER
#Defining two objects of open_dataset class:
train_ds=open_dataset(path2train_i, path2data_m=path2train_m, transform='train')
valid_ds=open_dataset(path2valid_i, path2data_m=path2valid_m, transform='val')
test_ds=open_dataset(path2test_i, path2data_m=path2test_m, transform='val')

#Create Data Loader
train_dl = DataLoader(train_ds, batch_size=3, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=3, shuffle=False) 
test_dl = DataLoader(test_ds, batch_size=3, shuffle=False) 

# tag_to_loc = {0 : ["train" , "test" , "val"] , 
#               1 : ["train1" , "test1" , "val1"] ,
#               2 : ["train2" , "test2" , "val2"]} 

# train_ds=Data_from_disk(typ = tag_to_loc[tag][0])
# valid_ds=Data_from_disk(typ = tag_to_loc[tag][2])
# test_ds=Data_from_disk(typ = tag_to_loc[tag][1])

# train_dl = DataLoader(train_ds, batch_size=3, shuffle=True)
# val_dl = DataLoader(valid_ds, batch_size=3, shuffle=False) 
# test_dl = DataLoader(test_ds, batch_size=3, shuffle=False) 

#model = UNet16(pretrained=True)
model = UNet()
ll = list(os.listdir("./cont"))
if len(ll) > 0 :
    model.load_state_dict(torch.load(ll[0]))

#Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f"Using device {device}")
if torch.cuda.is_available() :
    print("Using {}".format(torch.cuda.get_device_name(0)))
model=model.to(device)
#Print the mode
print(model)
# Show the model summary
summary(model, input_size=(3, 256, 256))
if torch.cuda.is_available() :
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

opt = optim.Adam(model.parameters(), lr=lr)
loss_func1 = nn.BCELoss()#nn.MSELoss()
loss_func2 = MALISLoss_window_pos()


path2models= "./models/"
params_train={
    "num_epochs": n_epoch,
    "optimizer": opt,
    "loss_func1": loss_func1, #mse
    "loss_func2": loss_func2, #malis - topo
    "train_dl": train_dl,
    "val_dl": val_dl,
    "test_dl": test_dl,
    "sanity_check": False,
    "path2weights": path2models+"_full_weights_{}_{}_{}_{}.pt".format(n_epoch, lr, malis_neg, malis_pos),
    "path2losshist": path2models+"_loss_history_{}_{}_{}_{}.json".format(n_epoch, lr, malis_neg, malis_pos),
    "malis_params": [malis_neg, malis_pos] ,
    "typ" : 2
}

model, loss_hist, test_loss = train_val(model,params_train)


