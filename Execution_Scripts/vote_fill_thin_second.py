import os
import numpy as np
# import imgaug as ia

from PIL import Image
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import thin


def vote_fill_thin_second(shift_root, No_shift_root, filled_dir, dir_second_vote, thinning_iter):
    
    img_folder = os.path.join(shift_root, filled_dir)
    img_names = os.listdir(img_folder)
    
    for img_name in img_names:
        mask_shift_full_dir = os.path.join(img_folder, img_name)
        mask_no_shift_full_dir = os.path.join(No_shift_root, filled_dir, img_name)
        
        mask_shift_temp = np.array(Image.open(mask_shift_full_dir))
        mask_no_shift_temp = np.array(Image.open(mask_no_shift_full_dir))
        
        
        row_len = len(mask_no_shift_temp[:,1])
        col_len = len(mask_no_shift_temp[1,:])
        
        mask_no_shift_tovote = mask_no_shift_temp[128:row_len-128, 128:col_len-128]
        mask_secondvoted = mask_shift_temp + mask_no_shift_tovote
        mask_secondvoted[np.where(mask_secondvoted>0)] = 255
        
        mask_secondvoted_filled = fill_second(mask_secondvoted)
        
        mask_second_filled_thin = border_thinning(mask_secondvoted_filled, thinning_iter)
        
        
        folder_tosave = os.path.join(shift_root, dir_second_vote)
        if not os.path.exists(folder_tosave):
            os.makedirs(folder_tosave)
        img_dir_tosave = os.path.join(folder_tosave, img_name)
        img_tosave = Image.fromarray(mask_second_filled_thin)    
        img_tosave.save(img_dir_tosave)
        
        # ia.imshow(mask_voted)
    
def border_thinning(image, thinning_iter):
    image = -(image/255-1)
    # thinning_iter = 6
    thinned_partial = image.copy()
    
    for i in range(1, thinning_iter):
        thinned_partial = thin(thinned_partial, max_iter=1)
        
        thinned_partial[0,:] = image[0,:]
        thinned_partial[len(image[:,0])-1, :] = image[len(image[:,0])-1, :]
        thinned_partial[:,0] = image[:,0]
        thinned_partial[:,len(image[0,:])-1] = image[:,len(image[0,:])-1]
    
    thinned_partial = np.uint8(-(thinned_partial-1)*255)
    
    return thinned_partial
    
def fill_second(mask_input):
    
    # Fill holes
    mask_input = mask_input / 255
    
    mask_hole_label = label(mask_input, connectivity=1)
    
    
    #Fill edges
    up_edge = []
    left_edge = []
    right_edge = []
    low_edge = []
    
    
    
    for i in range(1,len(mask_hole_label[1,:])-1):
        if (mask_input[0,i] != 0) and ((mask_hole_label[0,i] != mask_hole_label[0,i-1]) or (mask_hole_label[0,i] != mask_hole_label[0,i+1])):
            up_edge.append(i)        
        if (mask_input[len(mask_input[:,1])-1,i] != 0 ) and ((mask_hole_label[len(mask_input[:,1])-1,i] != mask_hole_label[len(mask_input[:,1])-1,i-1]) or (mask_hole_label[len(mask_input[:,1])-1,i] != mask_hole_label[len(mask_input[:,1])-1,i+1])):
            low_edge.append(i)
            
    for i in range(1,len(mask_hole_label[:,1])-1):    
        if (mask_input[i,0] != 0) and ((mask_hole_label[i,0] != mask_hole_label[i-1,0]) or (mask_hole_label[i,0] != mask_hole_label[i+1,0])):
            left_edge.append(i)
        if (mask_input[i,len(mask_input[1,:])-1] != 0) and ((mask_hole_label[i,len(mask_input[1,:])-1] != mask_hole_label[i-1,len(mask_input[1,:])-1]) or (mask_hole_label[i,len(mask_input[1,:])-1] != mask_hole_label[i+1,len(mask_input[1,:])-1])):
            right_edge.append(i)
    


    
    for i in range(len(up_edge)):
        for j in range(i+1, len(up_edge)):
            if mask_hole_label[0,up_edge[i]] == mask_hole_label[0,up_edge[j]]:
                mask_input[0, up_edge[i]:up_edge[j]] = 1
                a = 1
    for i in range(len(low_edge)):
        for j in range(i+1, len(low_edge)):
            if mask_hole_label[len(mask_input[:,1])-1,low_edge[i]] == mask_hole_label[len(mask_input[:,1])-1,low_edge[j]]:
                mask_input[len(mask_input[:,1])-1, low_edge[i]:low_edge[j]] = 1
    
    for i in range(len(left_edge)):
        for j in range(i+1, len(left_edge)):
            if mask_hole_label[left_edge[i],0] == mask_hole_label[left_edge[j],0]:
                mask_input[left_edge[i]:left_edge[j], 0] = 1
    for i in range(len(right_edge)):
        for j in range(i+1, len(right_edge)):
            if mask_hole_label[right_edge[i],len(mask_input[1,:])-1] == mask_hole_label[right_edge[j],len(mask_input[1,:])-1]:
                mask_input[right_edge[i]:right_edge[j], len(mask_input[1,:])-1] = 1            
    
    
    mask_hole_filled = ndimage.binary_fill_holes(mask_input).astype(int)
    mask_hole_filled = np.uint8(255 * mask_hole_filled)
    
    return mask_hole_filled