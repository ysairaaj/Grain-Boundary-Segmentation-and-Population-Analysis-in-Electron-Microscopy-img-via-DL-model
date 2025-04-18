import os
import PIL 
import PIL.Image
import warnings
import imgaug.augmenters as iaa
import numpy as np
from img_scale_trans import Scaling

warnings.filterwarnings('ignore')

def expand_by_mirroring(image_train_raw,
                        len_row, 
                        len_col,
                        half_len): #half_len = 512//4

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

def img_aug(img ,img_name, step_x , step_y ,  dim_x ,dim_y , save_loc = None , img_arr = []):
    c = 0

    if type(img) != "numpy.ndarray" :
        img = np.array(img)

    for i in range(0 ,img.shape[0] , step_x):
        if i + 2*dim_x > img.shape[0]:
            break
        for j in range(0 , img.shape[1] , step_y):
            if j + 2*dim_y > img.shape[1] :
                break 
            new_img = img[i : i + 2*dim_x , j : j + 2*dim_y ,:].copy()
            new_img = PIL.Image.fromarray(new_img)
            new_img = Scaling.preprocess_2(new_img, 0.5)
            #new_img = PIL.Image.fromarray(new_img)
            if save_loc == None :
                img_arr.append((new_img ,img_name + f"_{i}_{j}"))
            else:    
                new_img.save(save_loc + img_name + f"_{i}_{j}" + ".png")
            c+=1
    print("Done !!")
    print(f"Created {c} augmented images of {img_name} of size : {(dim_x ,dim_y)} taking the step size : {(step_x ,step_y)}")

def data_gen(raw_loc = "./data/Raw/img0/" , raw_loc_out = "./data/Raw/img0_out/" , out_loc = "./data/" , dim = (512 , 512) , data_ratio = (0.7,0.2,0.1) , save_data = True):
    
    aug_list = ['rt00', 'rt90', 'rt180', 'fplr', 'fpud'] 
    #ra = [None]
    def store(img,name, ind,tot  , out_train , out_val , out_test ):
        ra = ind/tot
        count = -1
        if ra < data_ratio[0] :
            out_dir = out_train 
            count = 0
        elif ra > data_ratio[0] + data_ratio[1] :
            out_dir = out_test
            count = 2
        else :
            count = 1
            out_dir = out_val 

        img.save(out_dir + name + ".png") 
        assert count != -1 , "{} did not get saved".format(name)
        return count
    
    def check_dirs(path):
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print(f"The new directory {path} is created!")
    
    ##for input data
    imgs = sorted(os.listdir(raw_loc))
    #ensures same dataset across same different devices 
    out_train = out_loc + "_inp_/" + "train/"
    out_val = out_loc + "_inp_/" + "val/"
    out_test = out_loc + "_inp_/" + "test/" 
    check_dirs(out_train)
    check_dirs(out_val)
    check_dirs(out_test)
    
    
    

    
    for aug_name in aug_list :
        for img in imgs :
            img_arr = []
            count = [0 , 0 ,0]
            new_img = np.array(PIL.Image.open(raw_loc + img))[:,:,:3]
            new_img = augmt5_0325(new_img , aug_name)
            shape = new_img.shape
            new_new_img = PIL.Image.fromarray(expand_by_mirroring(new_img , shape[0] , shape[1] ,dim[0]//4))
            img_aug(new_new_img , img.split(".")[0] + aug_name , 100 ,100 , dim[0],dim[1] , None , img_arr) 
            print("stored : " , len(img_arr))
            if save_data == True :
                for  i , img_with_name in enumerate(img_arr) :
                    img = img_with_name[0]
                    name = img_with_name[1] 
                    cc = store(img , name , i , len(img_arr) , out_train , out_val , out_test)  
                    count[cc] += 1   
                print("Distributed in train :{} , val :{} , test :{} ".format(count[0],count[1],count[2]))    

    print(f"Created {len(os.listdir(out_train))} train_set_inp , {len(os.listdir(out_val))} val_set_inp , {len(os.listdir(out_test))} test_set_inp .")  

    ##for output data 
    imgs = sorted(os.listdir(raw_loc_out  )) 
    out_train = out_loc + "_out_/" + "train/"
    out_val = out_loc + "_out_/" + "val/"
    out_test = out_loc + "_out_/" + "test/"
    check_dirs(out_train)
    check_dirs(out_val)
    check_dirs(out_test)


    
    for aug_name in aug_list :
        for img in imgs :
            img_arr = []
            count = [0 , 0 ,0]
            new_img = np.array(PIL.Image.open(raw_loc_out + img))[:,:,:3]
            new_img = augmt5_0325(new_img , aug_name)
            shape = new_img.shape
            new_new_img = PIL.Image.fromarray(expand_by_mirroring(new_img , shape[0] , shape[1] ,dim[0]//4))
            img_aug(new_new_img , img.split(".")[0] + aug_name , 100 ,100 , dim[0],dim[1] , None , img_arr) 
            print("stored : " , len(img_arr))
            if save_data == True :
                for  i , img_with_name in enumerate(img_arr) :
                    img = img_with_name[0]
                    name = img_with_name[1] 
                    cc = store(img , name , i , len(img_arr) , out_train , out_val , out_test)  
                    count[cc] += 1
                print("Distributed in train :{} , val :{} , test :{} ".format(count[0],count[1],count[2]))    

    print(f"Created {len(os.listdir(out_train))} train_set_out , {len(os.listdir(out_val))} val_set_out , {len(os.listdir(out_test))} test_set_out .")  

    return       












            


                   

    
    
