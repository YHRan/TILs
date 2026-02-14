import os
import glob
from PIL import Image
import numpy as np
import cv2


# s1
def s2i(semantic_mask_path, instance_save_dir):
    name = semantic_mask_path.split('/')[-1].split('.')[0]
    # 读取语义 mask
    semantic_mask = cv2.imread(semantic_mask_path, cv2.IMREAD_GRAYSCALE)

    # 获取语义 mask 的轮廓
    contours, _ = cv2.findContours(semantic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    n = len(contours)

    # 创建实例 mask 目录
    os.makedirs(instance_save_dir, exist_ok=True)
    cur_save_dir = os.path.join(instance_save_dir, name)
    os.makedirs(cur_save_dir, exist_ok=True)

    # 提取每个轮廓并保存为单独的 mask 图像
    if n != 0:
        for idx, contour in enumerate(contours):
            # 创建空白的实例 mask
            instance_mask = np.zeros_like(semantic_mask, dtype=np.uint8)
            # 在实例 mask 上绘制当前轮廓
            cv2.drawContours(instance_mask, [contour], -1, 255, thickness=cv2.FILLED)
            instance_mask_path = os.path.join(cur_save_dir, f'{idx + 1}.png')
            # 保存实例 mask
            cv2.imwrite(instance_mask_path, instance_mask)
    else:
        cv2.imwrite(os.path.join(cur_save_dir, '1.png'), np.zeros_like(semantic_mask, dtype=np.uint8))

def Boundary_contour(mask_path, new_mask_path):
    file = os.listdir(mask_path)
    img_path = []
    for i in file:
        print(i)
        file_path = os.path.join(mask_path, i)
        img = cv2.imread(file_path)  # 原图
        im = img.copy()  # 将原图copy一份
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            edge_margin = 3
            im[:, :edge_margin] = 0
            im[:edge_margin, :] = 0
            im[:, -edge_margin:] = 0
            im[-edge_margin:, :] = 0

            difference = cv2.subtract(img, im)
            result = not np.any(difference)
            if result == False and area <= 2000:
                # img_path.append(i)
                src = os.path.join(mask_path, i)
                r_name = i.split('.')[0] + '.png'
                dct = os.path.join(new_mask_path, r_name)
                os.rename(src, dct)


if __name__ == '__main__':
    img_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/img'
    semantic_mask_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/mask'

    instance_save_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/instance'
    save_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/instance_filtered_out'
    mask_processed_save_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/mask_processed'
    final_save_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/mask_processed_img'
    
    all_instance_mask_path = glob.glob(instance_save_dir + '/*')
    all_img_path = glob.glob(os.path.join(img_dir, '*.png'))

    os.makedirs(final_save_dir, exist_ok=True)
    os.makedirs(mask_processed_save_dir, exist_ok=True)
    os.makedirs(instance_save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    all_semantic_mask_path = glob.glob(semantic_mask_dir + '/*.png')
    print(len(all_semantic_mask_path))
    
    for semantic_mask_path in all_semantic_mask_path:
        s2i(semantic_mask_path, instance_save_dir)
    for instance_mask_path in all_instance_mask_path:
        name = instance_mask_path.split('/')[-1]
        new_mask_path = os.path.join(save_dir, name)
        os.makedirs(new_mask_path, exist_ok=True)
        Boundary_contour(instance_mask_path, new_mask_path)
        all_instance_masks = glob.glob(os.path.join(instance_mask_path, '*.png'))
        # patch_size=640
        mask = np.zeros((640, 640))
        for instance_mask in all_instance_masks:
            ins_mask = np.array(Image.open(instance_mask))
            mask[ins_mask != 0] = 255
        cv2.imwrite(os.path.join(mask_processed_save_dir, name + '.png'), mask)
    for img_path in all_img_path:
        img_name = img_path.split('/')[-1].split('.')[0]
        outer_instance_path = os.path.join(save_dir, img_name)
        all_outer_instance_masks = glob.glob(os.path.join(outer_instance_path, '*.png'))
        img = np.array(Image.open(img_path))
        for outer_mask_path in all_outer_instance_masks:
            outer_mask = cv2.imread(outer_mask_path)[:, :, 0]
            print(outer_mask.shape, np.unique(outer_mask))
            img[outer_mask != 0] = 0
        mask_list = os.listdir(mask_processed_save_dir)
        if img_name + '.png' in mask_list:
            cv2.imwrite(os.path.join(final_save_dir, img_name + '.png'), img[:, :, ::-1])
