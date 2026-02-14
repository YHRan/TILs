import cv2
import numpy as np
import os
import glob

# img_dir=r'D:\ProjectSpace\STRUCTURE_DEV_DATA_CLEANING\TE_shengjingxibao_zhichi_BUBIAO\xiuzheng\patch_data_train_valid\cls1_mask2labelme\img-bowl'
# semantic_mask_dir=r'D:\ProjectSpace\STRUCTURE_DEV_DATA_CLEANING\TE_shengjingxibao_zhichi_BUBIAO\xiuzheng\patch_data_train_valid\mask_vis\cls2'
# instance_save_dir=r'D:\ProjectSpace\STRUCTURE_DEV_DATA_CLEANING\TE_shengjingxibao_zhichi_BUBIAO\xiuzheng\patch_data_train_valid\zhichi_process\instance'
img_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/img'
semantic_mask_dir = '/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/mask'
instance_save_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/instance'


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
#             area = cv2.contourArea(contour)
            # 创建空白的实例 mask
            instance_mask = np.zeros_like(semantic_mask, dtype=np.uint8)
            # 在实例 mask 上绘制当前轮廓
            cv2.drawContours(instance_mask, [contour], -1, 255, thickness=cv2.FILLED)

            instance_mask_path = os.path.join(cur_save_dir, f'{idx + 1}.png')

            # 保存实例 mask
            cv2.imwrite(instance_mask_path, instance_mask)
    else:
        cv2.imwrite(os.path.join(cur_save_dir, '1.png'), np.zeros_like(semantic_mask, dtype=np.uint8))


all_semantic_mask_path = glob.glob(semantic_mask_dir + '/*.png')
print(len(all_semantic_mask_path))
for semantic_mask_path in all_semantic_mask_path:
    # print(semantic_mask_path)
    s2i(semantic_mask_path, instance_save_dir)
