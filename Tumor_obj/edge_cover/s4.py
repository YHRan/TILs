import os
import glob
import cv2
from PIL import Image
import numpy as np

img_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/img'
all_img_path = glob.glob(os.path.join(img_dir, '*.png'))

outer_instance_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/instance_filtered_out'

mask_list = os.listdir(
    r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/mask_processed')
final_save_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/mask_processed_img'
os.makedirs(final_save_dir, exist_ok=True)
for img_path in all_img_path:
    img_name = img_path.split('/')[-1].split('.')[0]

    outer_instance_path = os.path.join(outer_instance_dir, img_name)

    all_outer_instance_masks = glob.glob(os.path.join(outer_instance_path, '*.png'))

    img = np.array(Image.open(img_path))

    for outer_mask_path in all_outer_instance_masks:
        outer_mask = cv2.imread(outer_mask_path)[:, :, 0]
        print(outer_mask.shape, np.unique(outer_mask))
        img[outer_mask != 0] = 0

    if img_name + '.png' in mask_list:
        cv2.imwrite(os.path.join(final_save_dir, img_name + '.png'), img[:, :, ::-1])
