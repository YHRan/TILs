import os
import glob
from PIL import Image
import numpy as np
import cv2

instance_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/instance'

all_instance_path = glob.glob(os.path.join(instance_dir, '*'))

mask_processed_save_dir = r'/home/zyj/SD/xianpao/data/pre_data/ds_json_reduce/mask_processed'
os.makedirs(mask_processed_save_dir, exist_ok=True)
for instance_path in all_instance_path:
    name = instance_path.split('/')[-1]
    all_instance_masks = glob.glob(os.path.join(instance_path, '*.png'))
    mask = np.zeros((640, 640))
    for instance_mask in all_instance_masks:
        ins_mask = np.array(Image.open(instance_mask))
        mask[ins_mask != 0] = 255
    cv2.imwrite(os.path.join(mask_processed_save_dir, name + '.png'), mask)
