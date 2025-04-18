import cv2
import os
import numpy as np 
from tqdm import tqdm

target_loc = "./prediction/mse+topo_highdisconn/2-Final_Results"
#target_loc = "./prediction/test"
imgs = os.listdir(target_loc)
#print(imgs)
save_loc = "./post_processed/mse+topo_high_disconn"
isExist = os.path.exists(save_loc)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(save_loc)

def calculate_slope(line_object):
    x_point1 = line_object[0]
    y_point1 = line_object[1]
    x_point2 = line_object[2]
    y_point2 = line_object[3]

    m = abs((y_point2 - y_point1) / (x_point2 - x_point1 + 0.1))
    m = float("{:.2f}".format(m))
    return m


for img in imgs:
    img10 = img
    #print(target_loc + img)
    img = cv2.imread(target_loc + '/' + img)
    #img = cv2.bitwise_not(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = img
    #d = cv2.ximgproc.createFastLineDetector()
    #lines = d.detect(gray)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=4)
    # iterator = tqdm(lines)
    # for current_line in iterator:
    #     current_slope = calculate_slope(current_line[0])

    #     for neighbor_line in lines:
    #         current_x1 = int(current_line[0][0])
    #         current_y1 = int(current_line[0][1])
    #         current_x2 = int(current_line[0][2])
    #         current_y2 = int(current_line[0][3])
    #         #print(neighbor_line)
    #         compare_lines = current_line == neighbor_line[0]
    #         #print(compare_lines)
    #         equal_arrays = compare_lines.all()
    #         #print(equal_arrays)

    #         if not equal_arrays:
    #             neighbor_slope = calculate_slope(neighbor_line[0])  

    #             if abs(current_slope - neighbor_slope) < 1e-3:
    #                 neighbor_x1 = int(neighbor_line[0][0])
    #                 neighbor_y1 = int(neighbor_line[0][1])
    #                 neighbor_x2 = int(neighbor_line[0][2])
    #                 neighbor_y2 = int(neighbor_line[0][3])

    #                 cv2.line(img,
    #                         pt1=(neighbor_x1, neighbor_y1),
    #                         pt2=(current_x2, current_y2),
    #                         color= (255,255,255),
    #                         thickness=3)
    #                 cv2.line(img,
    #                         pt1=(current_x1, current_y1),
    #                         pt2=(neighbor_x2, neighbor_y2),
    #                         color= (255,255,255),
    #                         thickness=3)     
    print(f"writing {img10}")
    cv2.imwrite(os.path.join(save_loc + "/" ,img10) , img_erosion)

print("Done!")