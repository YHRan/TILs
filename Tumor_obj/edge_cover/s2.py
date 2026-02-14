import cv2
import numpy as np
import os
import glob


def Boundary_contour(mask_path, new_mask_path):
    file = os.listdir(mask_path)
    img_path = []
    for i in file:
        print(i)
        file_path = os.path.join(mask_path, i)
        img = cv2.imread(file_path)  # 原图
#         print(img.shape)
        im = img.copy()  # 将原图copy一份
#         add
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#         th = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
        
            edge_margin = 3
            im[:, :edge_margin] = 0
            im[:edge_margin, :] = 0
            im[:, -edge_margin:] = 0
            im[-edge_margin:, :] = 0

            difference = cv2.subtract(img, im)
            # print(difference)
            result = not np.any(difference)
            # print(result)
            if result == False and area <= 2000:
                # img_path.append(i)
                src = os.path.join(mask_path, i)
                r_name = i.split('.')[0] + '.png'
                dct = os.path.join(new_mask_path, r_name)
                os.rename(src, dct)


instance_mask_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/instance'
all_instance_mask_path = glob.glob(instance_mask_dir + '/*')

save_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/instance_filtered_out'
for instance_mask_path in all_instance_mask_path:
    name = instance_mask_path.split('/')[-1]
    new_mask_path = os.path.join(save_dir, name)
    os.makedirs(new_mask_path, exist_ok=True)
    Boundary_contour(instance_mask_path, new_mask_path)
