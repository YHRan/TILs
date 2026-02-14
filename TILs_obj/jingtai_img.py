import cv2
import os
import torch
import numpy as np

# 定义保存结果的路径
output_image_path = '/home/lyb/data_yhr/03-yhr/External_project/yolov7-main/runs/detect/exp3/time_series_image.png'

# 初始化空白图像（大小可以根据你的图像尺寸进行调整）
image_height, image_width = 540, 960  # 假设图像大小为640x640，你可以调整此尺寸
blank_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # 白色背景

# 假设图像序列存储路径（你可以根据实际路径进行修改）
image_folder = '/home/lyb/data_yhr/03-yhr/External_project/yolov7-main/runs/detect/exp3/img'
txt_folder = '/home/lyb/data_yhr/03-yhr/External_project/yolov7-main/runs/detect/exp3/labels'

# 遍历图像序列
for i,img_data in enumerate(os.listdir(image_folder)):
    if img_data.endswith('.png'):
        print(img_data)
        name = img_data.split('.')[0]
        txt_name = f"{name}.txt"  # 对应的txt文件
        print(txt_name)
        # 读取对应的txt文件
        txt_path = os.path.join(txt_folder, txt_name)

        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # 读取图像，确保尺寸一致
        img = cv2.imread(os.path.join(image_folder, img_data))
        h, w, _ = img.shape
        gn = torch.tensor([w, h, w, h])  # 用于归一化的尺寸，假设图像是640x640
        top_left = (255,0,0)
        top_right = (0,255,0)
        down_left = (0,0,255)
        down_right = (0,255,255)
        
        # 遍历txt文件中的每个检测框
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])  # 类别
            x_center, y_center = float(parts[1]), float(parts[2])  # 中心点坐标（归一化）
            
            # 将归一化的中心点坐标还原到原始图像尺寸
            x_center *= w
            y_center *= h
            
            # 绘制中心点，使用红色小圆点表示
            cv2.circle(blank_image, (int(x_center), int(y_center)), radius=3, color=(0, 0, 255), thickness=-1)

# 保存最终的图像
cv2.imwrite(output_image_path, blank_image)

print(f"save as_ {output_image_path}")
