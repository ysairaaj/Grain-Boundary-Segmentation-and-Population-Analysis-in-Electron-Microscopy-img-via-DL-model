import numpy as np 
import PIL.Image 
import cv2 

class PatchStitch :
    def __init__(self):
        self.patch_path = {}
        self.img_dim = None
        self.dim_max_x = -1 
        self.dim_max_y = -1
        self.size = None
        self.name = None
    def patch(self , img , size , loc ,name):
        self.name = name
        self.size = size
        img = np.array(img)
        size_x , size_y = size[0] , size[1]
        img_x , img_y = img.shape[0] , img.shape[1]
        self.img_dim = (img_x , img_y)
        for i in range(0 , img_x , size_x):
            for j in range(0 , img_y , size_y):
                self.dim_max_x = max(self.dim_max_x , i) 
                self.dim_max_y = max(self.dim_max_y , j)
                new_loc = loc + '/' + str(i) + '_' + str(j) + '.png'
                self.patch_path[(i ,j)] = new_loc
                PIL.Image.fromarray(img[i:min(i+size_x, img_x) , j : min(j+size_y , img_y)]).save(new_loc) 

        print("patched")

        return         
    
    def stitch(self , loc):
        new_img = np.zeros((self.img_dim[0] , self.img_dim[1] , 3)) 
        size_x , size_y = self.size[0] , self.size[1]
        for i in range(0 , self.img_dim[0] , size_x):
            for j in range(0 ,self.img_dim[1] , size_y):
                new_img[i:min(i+size_x, self.img_dim[0]) , j : min(j+size_y , self.img_dim[1]) , 0] = np.array(PIL.Image.open(self.patch_path[(i , j)]))[:,:,0]
                new_img[i:min(i+size_x, self.img_dim[0]) , j : min(j+size_y , self.img_dim[1]) , 1] = np.array(PIL.Image.open(self.patch_path[(i , j)]))[:,:,1]
                new_img[i:min(i+size_x, self.img_dim[0]) , j : min(j+size_y , self.img_dim[1]) , 2] = np.array(PIL.Image.open(self.patch_path[(i , j)]))[:,:,2]
        self.final = new_img
        PIL.Image.fromarray(new_img.astype(np.uint8)).convert('RGB').save(loc + '/' + self.name + '.png')

        print('DONE')

        return 



