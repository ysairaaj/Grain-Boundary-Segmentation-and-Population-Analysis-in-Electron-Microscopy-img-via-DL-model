from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset


class Scaling(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=0.5, mask_suffix='_mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, raw_img, scale_factor): # raw_image -> PIL image file
        w_raw, h_raw = raw_img.size
        W_scale, H_scale = int(scale_factor * w_raw), int(scale_factor * h_raw)
        assert W_scale > 0 and H_scale > 0, 'Scale is too small'
        raw_img = raw_img.resize((W_scale, H_scale))
    
        img_arr = np.array(raw_img)
    
        if len(img_arr.shape) == 2:
            img_arr = np.expand_dims(img_arr, axis=2)
    
        
        img_trans = img_arr.transpose((2, 0, 1)) # Trans: HWC - CHW
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    
    @classmethod
    def preprocess_2(cls, raw_img, scale_factor): # raw_image -> PIL image file
        w_raw, h_raw = raw_img.size
        W_scale, H_scale = int(scale_factor * w_raw), int(scale_factor * h_raw)
        assert W_scale > 0 and H_scale > 0, 'Scale is too small'
        raw_img = raw_img.resize((W_scale, H_scale))
    
        

        return raw_img