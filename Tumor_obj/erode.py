import cv2
import os
from memory_profiler import profile

# @profile()
def batch_image_erosion(input_folder, output_folder, kernel_size, iterations):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = os.listdir(input_folder)

    # 创建腐蚀核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 对每张图像进行腐蚀并保存
    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        # 进行腐蚀操作
        eroded_image = cv2.erode(image, kernel, iterations=iterations)

        # 保存腐蚀后的图像
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, eroded_image)

        # print(f"Processed: {image_file}")


if __name__ == '__main__':
    # 调用函数对指定文件夹中的图像进行批量腐蚀
    input_folder = r"/home/zyj/SD/xianpao/data/pre_data/test"
    output_folder = r"/home/zyj/SD/xianpao/data/pre_data/svs"

    # 腐蚀的核的大小和迭代次数
    kernel_size = (13, 13)
    iterations = 1

    batch_image_erosion(input_folder, output_folder, kernel_size, iterations)
