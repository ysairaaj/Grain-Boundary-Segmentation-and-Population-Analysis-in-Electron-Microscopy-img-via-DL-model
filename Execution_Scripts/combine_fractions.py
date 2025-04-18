import numpy as np
import os
from tqdm.notebook import trange, tqdm
from PIL import Image
import imgaug.augmenters as iaa
from reverse import rev_rotate

def combine_fractions (aug_root, aug_list, len_sub_img, image_sizes_dict):

    iter1 = tqdm(enumerate(aug_list) , total = len(aug_list) , desc = "augmentations")
    for _ , aug_name in iter1:
        
        prediction_path = aug_root + '/' + aug_name + '_pred'
        
        dirs_full = os.listdir(os.path.join('./', prediction_path))
        dirs = os.listdir(os.path.join('./', prediction_path))
        
        for j in range(0,len(dirs)):
            dirs[j] = dirs[j][:-10]
        names_full_imgs = list(set(dirs))
        names_full_imgs = [x for x in names_full_imgs if len(x) > 0]
        
        #create a folder for the full images
        aug_full_name = aug_name + '_pred_Full'
        folder_full = os.path.join(aug_root, aug_full_name)
        if not os.path.exists(folder_full):
            os.makedirs(folder_full)
        
        for img_name in names_full_imgs:
            selected_name = [x for x in dirs_full if img_name in x]
            index_temp = np.zeros((len(selected_name),2))
            for i in range(0,len(selected_name)):
                #prediction
                index_temp[i][0] = selected_name[i][-9:-7]
                index_temp[i][1] = selected_name[i][-6:-4]
                #true mask
                # index_temp[i][0] = selected_name[i][-14:-12]
                # index_temp[i][1] = selected_name[i][-11:-9]
            fracs_row = np.int8(np.max(index_temp[:,0]))
            fracs_col = np.int8(np.max(index_temp[:,1]))
            len_half = np.int(len_sub_img / 4)
            len_img_row = 2 * len_half * fracs_row
            len_img_col = 2 * len_half * fracs_col
            
            # This combined image contains residue in right and button side            
            mask_combined = np.uint8(np.ones([len_img_row,len_img_col]))
            
            for i in range(0,len(selected_name)):
                row_start = int(i/fracs_col) * 2* len_half
                col_start = (i%fracs_col) * 2* len_half
                mask_dir_temp = os.path.join(prediction_path,selected_name[i])
                mask_temp = Image.open(mask_dir_temp)
                mask_temp = np.array(mask_temp)
                mask_valid = mask_temp[len_half:3*len_half,len_half:3*len_half]
                # rev_name = prediction_path[2:]
                # mask_temp = rev_rotate(mask_temp, rev_name)
                mask_combined[row_start:row_start+2*len_half, col_start: col_start+2*len_half] = mask_valid
                
            
            # np.where(mask_combined>200, mask_combined, 255)
            # np.where(mask_combined<=200, mask_combined, 0)
            mask_combined[np.where(mask_combined>200)] = 255
            mask_combined[np.where(mask_combined<=200)] = 0
            
            mask_combined = rev_rotate(mask_combined, aug_name)
            

            len_row_raw = image_sizes_dict[img_name][0]
            len_col_raw = image_sizes_dict[img_name][1]
            mask_combined = mask_combined[:len_row_raw,:len_col_raw]
            
            
            img_tosave = Image.fromarray(mask_combined)
            save_dir = os.path.join(folder_full, (img_name + '.png'))
            img_tosave.save(save_dir)
            # mask_combined
    
        
    
    
    
    
    