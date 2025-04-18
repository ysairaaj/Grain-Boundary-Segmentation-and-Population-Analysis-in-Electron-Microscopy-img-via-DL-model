import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as Funct
from PIL import Image
from torchvision import transforms
from tqdm.notebook import trange, tqdm
# from unet import UNet
from Convs_Unet import UNet
from img_scale_trans import Scaling
from Im_seg import Im_Seg
from final_model import Im_Seg_2
import torchvision.transforms as T

def predict_img(model,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5 ,
                typ = 1):
    model.eval()


    #img_scale_trans = Scaling.preprocess(full_img, scale_factor)
    if typ == 1 :
        img_scale_trans_0 = Scaling.preprocess(full_img[0], scale_factor)
        img_scale_trans_1 = Scaling.preprocess(full_img[1], scale_factor)
        img_scale_trans_2 = Scaling.preprocess(full_img[2], scale_factor)

    # img = torch.from_numpy(img_scale_trans)   

    # img = img.unsqueeze(0)
    # img = img.to(device=device, dtype=torch.float32)
    if typ == 1:
        img0 = torch.from_numpy(img_scale_trans_0) 
    else :
        img0 = full_img[0]      
    img0 = img0.unsqueeze(0)
    img0 = img0.to(device=device, dtype=torch.float32)

    if typ == 1:
        img1 = torch.from_numpy(img_scale_trans_1)   
    else :
        img1 = full_img[1]    
    img1 = img1.unsqueeze(0)
    img1 = img1.to(device=device, dtype=torch.float32)

    if typ == 1:
        img2 = torch.from_numpy(img_scale_trans_2)
    else :
        img2 = full_img[2]       
    img2 = img2.unsqueeze(0)
    img2 = img2.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(img0 , img1 , img2)
        probs = output
        # if model.n_classes > 1:
        #     probs = Funct.softmax(output, dim=1)
        # else:
        #     probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        if typ == 1:
            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(full_img[0].size[1]),
                    transforms.ToTensor()
                ]
            )
            
        else :
            tf = transforms.Compose(
                [
                    #transforms.ToPILImage(),
                    transforms.Resize(512),
                ]
            )  

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def predict_tiles(dir_model,#'./Tiles_512_Sigmod6CT_ep150.pth'
                   tile_root,#'/prediction/example/1-Mid_Results'
                     aug_list , typ):#['rt00', 'rt90', 'rt180', 'fplr', 'fpud']
    
    '''
    output dirs will be 'aug_dirs' + '_pred'
    
    '''    

    scale = 0.5
    #mask_threshold = 0.5
    if typ == 1 :
        mask_threshold = 0.9
    else :
        mask_threshold = 8 
    iter1 = tqdm(enumerate(aug_list) , total = len(aug_list) , desc = "augmentations")
    for _ ,aug_name in iter1:
        
        aug_dir = os.path.join(tile_root, aug_name)
        aug_dir_0 = os.path.join(aug_dir ,'0')
        aug_dir_1 = os.path.join(aug_dir ,'1')
        aug_dir_2 = os.path.join(aug_dir ,'2')
        #in_files = os.listdir(aug_dir) 
        in_files_0 = os.listdir(aug_dir_0) 
        in_files_1 = os.listdir(aug_dir_1) 
        in_files_2 = os.listdir(aug_dir_2) 
        #out_dir = aug_dir + '_pred'
        out_dir = aug_dir + '_pred'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        in_files_0.sort()
        in_files_1.sort()
        in_files_2.sort()

        #out_files = in_files
        out_files_0 = in_files_0
        out_files_1 = in_files_1 
        out_files_2 = in_files_2 
        #model = UNet(n_channels=3, n_classes=1)
        
        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        if typ == 1:
            model = Im_Seg(device , loc = 'Models/') 
        else :
            model = Im_Seg_2(device , loc = 'Models/')    
        logging.info("Loading model")
        model.to(device=device)
        model.load_state_dict(torch.load(dir_model, map_location=device))
    
        logging.info("Model loaded !")
        iter2 = tqdm(enumerate(in_files_0) , total = len(in_files_0) , desc = "Num_Image")
        for i, fn in iter2:
            # logging.info("\nPredicting image {} ...".format(fn))
    
            #img = Image.open(os.path.join(aug_dir, fn))
            img0 = Image.open(os.path.join(aug_dir_0, fn))
            img1 = Image.open(os.path.join(aug_dir_1, fn))
            img2 = Image.open(os.path.join(aug_dir_2, fn))
            if typ != 1 :
                img0 = img0.convert('RGB')
                img1 = img1.convert('RGB')
                img2 = img2.convert('RGB')
                im_size = 256
                trans = T.Resize((im_size,im_size))
                img0 = trans(img0)
                img1 = trans(img1)
                img2 = trans(img2)
                trans = T.ToTensor()
                img0 = trans(img0)
                img1 = trans(img1)
                img2 = trans(img2)
                meanR, meanG, meanB = .485,.456,.406
                stdR, stdG, stdB = .229, .224, .225 
                norm_= T.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
                img0 = norm_(img0)
                img1 = norm_(img1)
                img2 = norm_(img2)
            mask = predict_img(model = model,
                               full_img = (img0 , img1 , img2),
                               scale_factor = scale,
                               out_threshold = mask_threshold,
                               device = device , typ = typ)
    

            out_fn = out_files_0[i]
            result = mask_to_image(mask)
            result.save(os.path.join(out_dir, out_fn))

            # logging.info("Mask saved to {}".format(out_files[i]))
    
