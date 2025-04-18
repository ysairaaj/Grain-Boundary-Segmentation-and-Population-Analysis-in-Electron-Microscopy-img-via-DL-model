import os
import numpy as np

from PIL import Image, ImageFilter
import imgaug.augmenters as iaa
from tqdm.notebook import trange, tqdm
np.int = int

def topredict_filter_Mirror_Aug_overtiles(raw_folder,#  /data/example/
                      fullimg_dir,#/prediction/example/0-Full/Full_raw/
                      augfull_dir,#'/prediction/example/0-Full/Full_aug/'
                      tiles_dir,#'/prediction/example/1-Mid_Results' 
                      aug_list,#['rt00', 'rt90', 'rt180', 'fplr', 'fpud']
                      len_tile,#512
                      img_name):#'img.jpg'
    dirs = os.listdir(raw_folder)
    
    image_sizes_dict = {}
    
    for i in trange(len(dirs) , total = len(dirs) , desc = "Num_Images"):
        run_temp = dirs[i]
        #print(run_temp , os.path.join(raw_folder, run_temp, img_name.split('.')[0] + '0'+'.' + img_name.split('.')[1]) , Image.open(os.path.join(raw_folder, run_temp, img_name.split('.')[0] + '0'+'.' + img_name.split('.')[1])))
        ##image_train_raw0 = np.array(Image.open(os.path.join(raw_folder, run_temp, img_name)))
        image_train_raw_0 = np.array(Image.open(os.path.join(raw_folder, run_temp, img_name.split('.')[0] + '0'+'.' + img_name.split('.')[1])))
        image_train_raw_1 = np.array(Image.open(os.path.join(raw_folder, run_temp, img_name.split('.')[0] + '1'+'.' + img_name.split('.')[1])))
        image_train_raw_2 = np.array(Image.open(os.path.join(raw_folder, run_temp, img_name.split('.')[0] + '2'+'.' + img_name.split('.')[1])))
        #print(image_train_raw_0.shape)
        if img_name.split('.')[1] != "" :
           image_train_raw_0 = image_train_raw_0[:,:,:3] 
           image_train_raw_1 = image_train_raw_1[:,:,:3]
           image_train_raw_2 = image_train_raw_2[:,:,:3]
        #Filter
        # image_train_raw = np.array(image_train_raw0.filter(ImageFilter.DETAIL))     
        ##image_train_raw = iaa.SigmoidContrast(gain=6)(image = image_train_raw0)
        image_train_raw_0 = iaa.SigmoidContrast(gain=6)(image = image_train_raw_0)
        image_train_raw_1 = iaa.SigmoidContrast(gain=6)(image = image_train_raw_1)
        image_train_raw_2 = iaa.SigmoidContrast(gain=6)(image = image_train_raw_2)
        # image_train_raw = np.array(image_train_raw0.filter(ImageFilter.EDGE_ENHANCE))   
        # Usually
        # mask_train_raw = np.array(Image.open(os.path.join(raw_folder, run_temp, 'mask.png')))

        
        # Expand_overlaptile_Mirroring
        # len_row_raw = len(image_train_raw[:,1,1])
        # len_col_raw = len(image_train_raw[1,:,1])
        # half_len = np.int(len_tile/4)
        # image_to_train= expand_by_mirroring(image_train_raw, len_row_raw, len_col_raw, half_len)          
        # save_full_raw(image_train_raw, run_temp, fullimg_dir) # Optional
        len_row_raw_0 = len(image_train_raw_0[:,1,1])
        len_col_raw_0 = len(image_train_raw_0[1,:,1])
        len_row_raw_1 = len(image_train_raw_1[:,1,1])
        len_col_raw_1 = len(image_train_raw_1[1,:,1])
        len_row_raw_2 = len(image_train_raw_2[:,1,1])
        len_col_raw_2 = len(image_train_raw_2[1,:,1])
        half_len = np.int(len_tile/4)
        image_to_train_0= expand_by_mirroring(image_train_raw_0, len_row_raw_0, len_col_raw_0, half_len)
        image_to_train_1= expand_by_mirroring(image_train_raw_1, len_row_raw_1, len_col_raw_1, half_len)
        image_to_train_2= expand_by_mirroring(image_train_raw_2, len_row_raw_2, len_col_raw_2, half_len)
        save_full_raw(image_train_raw_0 ,image_train_raw_1 ,image_train_raw_2 , run_temp, fullimg_dir) # Optional
        
        
        
        
        # Augumatation Six times
        
        for aug_name in aug_list:
            #image_to_train_aug = augmt5_0325(image_to_train, aug_name)
            image_to_train_aug_0 = augmt5_0325(image_to_train_0, aug_name)
            image_to_train_aug_1 = augmt5_0325(image_to_train_1, aug_name)
            image_to_train_aug_2 = augmt5_0325(image_to_train_2, aug_name)
            save_aug(image_to_train_aug_0, image_to_train_aug_1, image_to_train_aug_2  , run_temp, augfull_dir, aug_name)  # Optional
            create_overlaptiles(image_to_train_aug_0,image_to_train_aug_1,image_to_train_aug_2, run_temp, aug_name, tiles_dir, len_tile)  # Predict
        
        #size = [len_row_raw, len_col_raw]
        size_0 = [len_row_raw_0, len_col_raw_0]
        size_1 = [len_row_raw_1, len_col_raw_1]   
        size_2 = [len_row_raw_2, len_col_raw_2]
        image_sizes_dict[run_temp] = [size_0 , size_1 ,size_2]      #size
        
    return image_sizes_dict
        
        
            

#%% expand and save
def expand_by_mirroring(image_train_raw,
                        len_row, 
                        len_col,
                        half_len):

    """ 
# For testing,
    1. expand left and up side by GLOBAL['len_Unet_tile']/4.
    2. expand right and button side by "residue + GLOBAL['len_Unet_tile']/4", where residue equals "GLOBAL['len_Unet_tile'] - len_row%GLOBAL['len_Unet_tile']"

"""
    len_img_tile = 4*half_len
    if len_row%len_img_tile == 0:
        row_expand = np.int(len_row + 0.5 * len_img_tile)
        col_expand = np.int(len_col + 0.5 * len_img_tile)        
    else:
        row_expand = np.int(len_row + 1.5 * len_img_tile - len_row%len_img_tile)
        col_expand = np.int(len_col + 1.5 * len_img_tile - len_col%len_img_tile)
    
    img_expand = np.uint8(np.zeros([row_expand, col_expand, 3]))

    

    if len_row%len_img_tile == 0:
        img_expand[half_len:half_len + len_row, half_len : half_len + len_col, :] = image_train_raw  
        img_expand[:half_len, :, :] = np.flipud(img_expand[half_len : 2 * half_len, :, :])
        img_expand[half_len + len_row:row_expand, :, :] = np.flipud(img_expand[half_len + len_row - half_len:half_len + len_row, :, :])
        img_expand[:, :half_len, :] = np.fliplr(img_expand[:, half_len :  2 * half_len, :])
        img_expand[:, half_len + len_col:col_expand, :] = np.fliplr(img_expand[:, half_len + len_col - half_len:half_len + len_col, :])    
         
    else:
        img_expand[half_len:half_len + len_row, half_len : half_len + len_col, :] = image_train_raw  
        img_expand[:half_len, :, :] = np.flipud(img_expand[half_len : 2 * half_len, :, :])
        img_expand[half_len + len_row:row_expand, :, :] = np.flipud(img_expand[half_len + len_row - (5 * half_len - len_row%len_img_tile):half_len + len_row, :, :])
        img_expand[:, :half_len, :] = np.fliplr(img_expand[:, half_len :  2 * half_len, :])
        img_expand[:, half_len + len_col:col_expand, :] = np.fliplr(img_expand[:, half_len + len_col - (5 * half_len - len_col%len_img_tile):half_len + len_col, :])    
        
    return img_expand



def save_full_raw(image_to_train_0 ,image_to_train_1,image_to_train_2 , run_temp, fullimg_dir):
    #img_full_folder = os.path.join(fullimg_dir, 'img')
    img_full_folder_0 = os.path.join(fullimg_dir, 'img0')
    img_full_folder_1 = os.path.join(fullimg_dir, 'img1')
    img_full_folder_2 = os.path.join(fullimg_dir, 'img2')

    # if not os.path.exists(img_full_folder):
    #     os.makedirs(img_full_folder)
    if not os.path.exists(img_full_folder_0):
         os.makedirs(img_full_folder_0)
    if not os.path.exists(img_full_folder_1):
         os.makedirs(img_full_folder_1)
    if not os.path.exists(img_full_folder_2):
         os.makedirs(img_full_folder_2)    

    img_name = run_temp + '.png'

    
    #img_path = os.path.join(img_full_folder, img_name)
    img_path_0 = os.path.join(img_full_folder_0, img_name)
    img_path_1 = os.path.join(img_full_folder_1, img_name)
    img_path_2 = os.path.join(img_full_folder_2, img_name)
    #img_tosave = Image.fromarray(image_to_train)
    img_tosave_0 = Image.fromarray(image_to_train_0)
    img_tosave_1 = Image.fromarray(image_to_train_1)
    img_tosave_2 = Image.fromarray(image_to_train_2)
    #img_tosave.save(img_path)
    img_tosave_0.save(img_path_0)
    img_tosave_1.save(img_path_1)
    img_tosave_2.save(img_path_2)



#%% save_aug
def save_aug(image_to_train_0 ,image_to_train_1,image_to_train_2 , run_temp, fullimg_dir, aug_name):
    #img_full_folder = os.path.join(fullimg_dir, 'img')
    img_full_folder_0 = os.path.join(fullimg_dir, 'img0')
    img_full_folder_1 = os.path.join(fullimg_dir, 'img1')
    img_full_folder_2 = os.path.join(fullimg_dir, 'img2')

    # if not os.path.exists(img_full_folder):
    #     os.makedirs(img_full_folder)
    if not os.path.exists(img_full_folder_0):
        os.makedirs(img_full_folder_0)
    if not os.path.exists(img_full_folder_1):
        os.makedirs(img_full_folder_1)   
    if not os.path.exists(img_full_folder_2):
        os.makedirs(img_full_folder_2)     

   
    img_name = run_temp + '_' + aug_name + '.png'

    
    #img_path = os.path.join(img_full_folder, img_name)
    img_path_0 = os.path.join(img_full_folder_0, img_name)
    img_path_1 = os.path.join(img_full_folder_1, img_name)
    img_path_2 = os.path.join(img_full_folder_2, img_name)
    #img_tosave = Image.fromarray(image_to_train)
    img_tosave_0 = Image.fromarray(image_to_train_0)
    img_tosave_1 = Image.fromarray(image_to_train_1)
    img_tosave_2 = Image.fromarray(image_to_train_2)
    #img_tosave.save(img_path)
    img_tosave_0.save(img_path_0)
    img_tosave_1.save(img_path_1)
    img_tosave_2.save(img_path_2)


#%% aug_5
def augmt5_0325(img_expand,
                aug_method):

    aug_dict = {"rt00":rt00(img_expand),
                "rt90":rt90(img_expand),
                "rt180":rt180(img_expand),
                "fplr":fplr(img_expand), 
                "fpud":fpud(img_expand)}
    image = aug_dict[aug_method]    
    
    return image

def rt00(img):
    img_aug = img
    return img_aug

def rt90(img):
    img_aug = np.rot90(img)
    return img_aug

def rt180(img):
    img_aug = np.rot90(img, 2)
    return img_aug

def fplr(img):
    img_aug = iaa.Fliplr(1.0)(image = img)
    return img_aug

def fpud(img):
    img_aug = iaa.Flipud(1.0)(image = img)
    return img_aug



#%% create_tiles



def create_overlaptiles(img_temp_0 ,img_temp_1,img_temp_2, img_name, aug_name, tiles_folder, len_tile):

    """
    For overlaptile: the size of every tiles are 256, but the moving distance between tiles are 128;
    And only the valid prediction of central 128*128 images are used finally.
    
    """
    
    sample_length = np.int(len_tile/2)

    # num_row = np.int8(len(img_temp[:,1,1]) / sample_length)-1
    # num_col = np.int8(len(img_temp[1,:,1]) / sample_length)-1
    num_row_0 = np.int8(len(img_temp_0[:,1,1]) / sample_length)-1
    num_col_0 = np.int8(len(img_temp_0[1,:,1]) / sample_length)-1
    num_row_1 = np.int8(len(img_temp_1[:,1,1]) / sample_length)-1
    num_col_1 = np.int8(len(img_temp_1[1,:,1]) / sample_length)-1
    num_row_2 = np.int8(len(img_temp_2[:,1,1]) / sample_length)-1
    num_col_2 = np.int8(len(img_temp_2[1,:,1]) / sample_length)-1
    
    sample_folder_img = tiles_folder + "/" + aug_name + "/"
    if not os.path.exists(sample_folder_img):
        os.makedirs(sample_folder_img)
    
    if not os.path.exists(sample_folder_img + '0/'):
        os.makedirs(sample_folder_img + '0/')
    if not os.path.exists(sample_folder_img + '1/'):
        os.makedirs(sample_folder_img + '1/')
    if not os.path.exists(sample_folder_img + '2/'):
        os.makedirs(sample_folder_img + '2/')    
    # for i in range(num_row):
    #     for j in range(num_col):

    #         sample_temp = img_temp[sample_length*i:sample_length*i + len_tile, sample_length*j : sample_length * j  + len_tile, :]
                
    #         if i + 1 < 10:
    #             num_r_str = '0' + str(i+1)
    #         else:
    #             num_r_str = str(i+1)
                
    #         if j + 1 < 10:
    #             num_c_str = '0' + str(j+1)
    #         else:
    #             num_c_str = str(j+1)
                

    #         write_dir_img = sample_folder_img + img_name + '_'  + num_r_str + '_' + num_c_str + '.jpg'
            
    #         img_tosave = Image.fromarray(sample_temp)
    #         img_tosave.save(write_dir_img)
    for i in range(num_row_0):
        for j in range(num_col_0):

            sample_temp_0 = img_temp_0[sample_length*i:sample_length*i + len_tile, sample_length*j : sample_length * j  + len_tile, :]
                
            if i + 1 < 10:
                num_r_str = '0' + str(i+1)
            else:
                num_r_str = str(i+1)
                
            if j + 1 < 10:
                num_c_str = '0' + str(j+1)
            else:
                num_c_str = str(j+1)
                

            write_dir_img_0 = sample_folder_img + '0/' + img_name + '_'  + num_r_str + '_' + num_c_str + '.png'
            
            img_tosave_0 = Image.fromarray(sample_temp_0)
            img_tosave_0.save(write_dir_img_0)

    for i in range(num_row_1):
        for j in range(num_col_1):

            sample_temp_1 = img_temp_1[sample_length*i:sample_length*i + len_tile, sample_length*j : sample_length * j  + len_tile, :]
                
            if i + 1 < 10:
                num_r_str = '0' + str(i+1)
            else:
                num_r_str = str(i+1)
                
            if j + 1 < 10:
                num_c_str = '0' + str(j+1)
            else:
                num_c_str = str(j+1)
                

            write_dir_img_1 = sample_folder_img + '1/' + img_name + '_'  + num_r_str + '_' + num_c_str + '.png'
            
            img_tosave_1 = Image.fromarray(sample_temp_1)
            img_tosave_1.save(write_dir_img_1)            

    for i in range(num_row_2):
        for j in range(num_col_2):

            sample_temp_2 = img_temp_2[sample_length*i:sample_length*i + len_tile, sample_length*j : sample_length * j  + len_tile, :]
                
            if i + 1 < 10:
                num_r_str = '0' + str(i+1)
            else:
                num_r_str = str(i+1)
                
            if j + 1 < 10:
                num_c_str = '0' + str(j+1)
            else:
                num_c_str = str(j+1)
                

            write_dir_img_2 = sample_folder_img + '2/' + img_name + '_'  + num_r_str + '_' + num_c_str + '.png'
            
            img_tosave_2 = Image.fromarray(sample_temp_2)
            img_tosave_2.save(write_dir_img_2)        
