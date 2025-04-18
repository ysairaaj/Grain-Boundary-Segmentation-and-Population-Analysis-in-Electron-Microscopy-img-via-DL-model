import os
import imgaug.augmenters as iaa
import numpy as np

from PIL import Image


def reverse_main(aug_root, aug_list):
    for i, aug_name in enumerate(aug_list):
        
        prediction_path = aug_root + '/' + aug_name + '_pred_Full'
        dirs = os.listdir(os.path.join('./', prediction_path))
        
        for j in range(0,len(dirs)):            
            run_temp = dirs[j]
            precition_path_mask = os.path.join(prediction_path, run_temp)
            mask = Image.open(precition_path_mask)
            mask_arr = np.array(mask)
            
            mask_rev = rev_rotate(mask_arr, aug_name)
            
            img_tosave = Image.fromarray(mask_rev)
            img_tosave.save(os.path.join(prediction_path, run_temp))
    
            
            
            
            
def rev_rotate(mask_temp, rev_name):
    rotate_dict = {"rt00":rev_rt00(mask_temp), "rt90":rev_rt90(mask_temp), "rt180":rev_rt180(mask_temp), "fplr":rev_fplr(mask_temp), "fpud":rev_fpud(mask_temp),}
    mask = rotate_dict[rev_name]
    return  mask
    
def rev_rt00(mask):
    return mask

def rev_rt90(mask):
    mask_rev = np.rot90(mask, 3)
    return mask_rev

def rev_rt180(mask):
    mask_rev = iaa.Affine(rotate = -180)(image=mask)
    return mask_rev

def rev_fplr(mask):
    mask_rev = iaa.Fliplr(1.0)(image=mask)
    return mask_rev

def rev_fpud(mask):
    mask_rev = iaa.Flipud(1.0)(image=mask)
    return mask_rev
    