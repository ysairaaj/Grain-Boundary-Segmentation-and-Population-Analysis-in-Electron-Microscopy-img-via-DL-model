import cv2 
import numpy as np
import os 
from tqdm import tqdm
from collections import deque
import math
from sklearn.cluster import KMeans
import PIL.Image


#get connected components from a masked image . Returns map of format => index : list of coordinates , color
def get_conn_comps(img , img_in):
    img = np.array(img)
    img_in = np.array(img_in)
    if len(img.shape) == 3:
        img = img[:,:,0]
    if img_in.shape[2] == 4:
        img_in = img_in[:,:,0:3]    
    print(img.shape)
    dim_x , dim_y = img.shape[0] , img.shape[1]
    count = 0
    dic = {}
    dir = [(0,1) , (0,-1) , (1,0) , (-1,0)] 
    vis = [[0]*dim_y for _ in range(dim_x)]

    def bfs(dic ,vis , start , count):
        Q = deque([start])

        while Q :
            x , y = Q.popleft()
            for x1 , y1 in dir :
                if x+x1 >= 0 and x+x1 < dim_x and y+y1 >= 0 and y+y1 < dim_y and vis[x+x1][y+y1] == 0 and img[x+x1 ,y+y1] > 0:
                    vis[x+x1][y+y1] = 1
                    dic[count].append(((x+x1 , y+y1) , img_in[x+x1 ,y+y1 , :]) )
                    #dic[count].append((None, img_in[x+x1 ,y+y1 , :]) )
                    Q.append((x+x1 ,y+y1))
        return             

    for i in range(dim_x):
        for j in range(dim_y):
            if vis[i][j] == 0 and img[i][j] > 0:
                vis[i][j] = 1
                count+=1 
                dic[count] = []
                dic[count].append(((i,j) , img_in[i ,j , :]))
                bfs(dic, vis , (i,j) , count)
    dic = {key : dic[key] for key in dic.keys() if float(len(dic[key]))/float((dim_x*dim_y)) >= 0.0001}
    return dic   

#gets vector median of indexes in form of map of format => index : list of coordinates , color . Returns map of format => (index , median color)
def get_vec_median(dic , approx = True):
    cc = dic.keys()
    def dist(val1 , val2):
        assert len(val1) == 3 and len(val2) == 3 , f"Expected dim = 3 got dim = {len(val1)} , {len(val2)} respectively ."
        return math.sqrt((val1[0] - val2[0])**2 + (val1[1] - val2[1])**2 + (val1[2] - val2[2])**2)
    
    def get_median_dist(lst):
        dist_val = []
        for i , val in enumerate(lst) :
            if approx == True and i%(len(lst)//10) != 0:
                continue
            val_coor = val[0]
            val_rgb = val[1]
            ss = 0
            for j , val2 in enumerate(lst):
                if i == j:
                    continue 
                val2_rgb = val2[1]
                ss += dist(val_rgb , val2_rgb)
            dist_val.append((float(ss)/float(len(lst)+0.01) , val_rgb))   

        dist_val.sort(key = lambda x : x[0])

        return dist_val[len(dist_val)//2][1]   
    map_ = []
    for key in cc :
        map_.append((key , get_median_dist(dic[key])))

    return map_    
                          


#get labels for each index , input => (index , color) , output => (index , label)
def get_clusters(map_ , k = 3 , fit_model = None):
    map_val = np.array([x[1] for x in map_])
    if fit_model == None :
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(map_val)
    else :
        kmeans = fit_model  
    labels = kmeans.predict(map_val)     
    ans = {map_[i][0] :labels[i]  for i in range(len(map_))}
    return ans


# colors image using calculated cluster labels for each connected component
def get_img_clusters(img , map_ , dic , colors = [[255 , 0 , 0] , [0,255,0] , [0 , 0 ,255]]):
    img = img.convert('RGB')
    img = np.array(img)
    if len(img.shape) == 2 or img.shape[2] == 1 :
        img1 = np.zeros((img.shape[0] , img.shape[1] , 3))
        img1[:,:,0] = img 
        img1[:,:,1] = img 
        img1[:,:,2] = img 
        img = img1 
        
    for key in dic.keys():
        for loc , color in dic[key] :
            loc1 , loc2 = loc[0] , loc[1]
            img[loc1 , loc2 , 0] = colors[map_[key]][0]
            img[loc1 , loc2 , 1] = colors[map_[key]][1]
            img[loc1 , loc2 , 2] = colors[map_[key]][2] 
    return PIL.Image.fromarray(img) 

# gets centroid for each connected component , input => index : list of coordinates , output => index : centroid coordinate
def get_centroids(dic):

    def cent(lst):
        x_cent = [lst[i][0] for i in range(len(lst))]
        y_cent = [lst[i][1] for i in range(len(lst))]  

        return np.mean(x_cent) , np.mean(y_cent)    
    new_dic = {}
    for key in dic.keys() :
        lst = dic[key]
        new_dic[key] = cent(lst)

    return new_dic     


          
        
