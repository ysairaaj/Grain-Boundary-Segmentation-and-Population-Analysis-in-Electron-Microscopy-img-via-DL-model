
import os
import numpy as np


from PIL import Image
from scipy import ndimage
from vote_fill_thin_second import fill_second


def ensemble_vote_fill(aug_root, prediction_final, aug_list, vote):
    #print("hey")
    
    folder_full = [0,0,0,0,0]
    for i, aug_name in enumerate(aug_list):
        aug_full_name = aug_name + '_pred_Full'
        folder_full[i] = os.path.join(aug_root, aug_full_name)
    
    dirs = os.listdir(folder_full[0])
    
    for mask_name in dirs:
        img_agus_list = [0,0,0,0,0]
        for i in range(len(aug_list)):          
            img_agus_list[i] = np.array(Image.open(os.path.join(folder_full[i], mask_name)))
            img_agus_list[i] = img_agus_list[i] == 255
            print(f"inserting {os.path.join(folder_full[i], mask_name)}")
        
        
        row_img = len(img_agus_list[0][:,0])
        col_img = len(img_agus_list[0][0,:])

        img_en = np.ones([row_img, col_img])
        img_en = np.uint8(img_en)
        
    
        thresh = len(aug_list) - vote
        #print(f"threshold vote : {thresh}")
        
        #Image_sum = img_agus_list[0] and img_agus_list[1] and img_agus_list[2] and img_agus_list[3] and img_agus_list[4]
        Image_sum = np.logical_and(np.logical_and(np.logical_and(img_agus_list[0] ,img_agus_list[1] ) , img_agus_list[2]) , np.logical_and(img_agus_list[3] , img_agus_list[4]))
        img_en = np.uint8(Image_sum)
        #img_en[np.where(Image_sum <= thresh)] = 0
        
        # print("change")
        # for i in range(row_img):
        #     for j in range(col_img):
        #         sum_pix = img_agus_list[0][i,j] and img_agus_list[1][i,j] and img_agus_list[2][i,j] and img_agus_list[3][i,j] and img_agus_list[4][i,j]
        #         #if sum_pix <= thresh:
        #         img_en[i,j] = sum_pix

        img_en = img_en * 255
        
        #mask_filled = fill_second(img_en)
        mask_filled = img_en
        img_tosave = Image.fromarray(mask_filled)
        aug_full_voted = prediction_final
        if not os.path.exists(aug_full_voted):
            os.makedirs(aug_full_voted)
            
        save_dir = os.path.join(aug_full_voted, mask_name)
        img_tosave.save(save_dir)
        
def fill_holes(mask_temp):
    

    mask_filled = ndimage.binary_fill_holes(mask_temp).astype(int)
    mask_filled = np.uint8(255 * mask_filled)
       
    return mask_filled
