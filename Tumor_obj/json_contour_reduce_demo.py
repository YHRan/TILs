import os
import json
import cv2
import numpy as np
from read_img import TiffImg
from tqdm import tqdm
import glob


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def cut(roi_index, roi_img_array, roi_mask_array, save_dir, save_name, cut_size, cut_stride):
    os.makedirs(os.path.join(save_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'mask'), exist_ok=True)

    height, width = roi_img_array.shape[0], roi_img_array.shape[1]

    cut_row_num = (height - cut_size) // cut_stride + 1
    cut_col_num = (width - cut_size) // cut_stride + 1
    image_num = cut_row_num * cut_col_num

    num = 0
    i, j = 0, 0

    while i <= roi_img_array.shape[0] - cut_size:
        while j <= roi_img_array.shape[1] - cut_size:
            patch_img = roi_img_array[i:i + cut_size, j:j + cut_size, :]
            patch_mask = roi_mask_array[i:i + cut_size, j:j + cut_size]

            # 构造保存名称
            cur_save_name = f"{save_name}_{roi_index}_{i}_{j}.png"
            # 保存图像块和掩码块
            img_area = patch_mask.shape[0] * patch_mask.shape[1]
            if np.count_nonzero(patch_mask) > 0.001 * img_area:
                cv2.imwrite(os.path.join(save_dir, 'img', cur_save_name), patch_img)
                cv2.imwrite(os.path.join(save_dir, 'mask', cur_save_name), patch_mask)

            j = j + cut_stride
            num += 1
        j = 0
        i = i + cut_stride


def process_one_slide(svs_path, json_path, level, anno_label_name, save_dir, cut_size, overlap, roi_label_name=None):
    '''
    svs_path: 切片路径
    json_path：标注数据(新格式)json文件路径
    level:下采样倍率
    anno_label_name：标注数据中的目标类别名字
    save_dir：切图后patch数据存储路径
    cut_size：patch尺寸
    overlap：叠切步长
    roi_label_name：区域标注时，标注数据中的roi矩形框对应的label_name
    '''

    save_name = svs_path.split(os.sep)[-1].split('.')[0]
    # 读取svs
    read_svs = TiffImg(svs_path)
    img_ds = read_svs.get_img_ds(level)
    img_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)
    h, w, _ = img_ds.shape
    mask_ds = np.zeros((h, w))

    # 解析标注的新格式json文件，并将轮廓绘制到mask画布上
    data = read_json(json_path)
    all_feas = data['features']
    n = len(all_feas)
    rois = []  # 区域标注时的roi矩形框
    cnt = 0
    for feature in tqdm(all_feas, total=n):
        cont_label_name = feature['properties']['label_name']

        # contour_coordinates = feature['geometry']['coordinates'][0]
        contour_coordinates = feature['geometry']
        erode_shapes(contour_coordinates, erosion_factor, h, w)

        # print('contour_coordinates=', contour_coordinates)
        contour_coordinates = contour_coordinates['coordinates']
        # print(contour_coordinates)

        contour_coordinates = np.array(contour_coordinates, dtype=np.int32) // level

        if cont_label_name == anno_label_name:
            cnt += 1
            cv2.drawContours(mask_ds, [contour_coordinates], -1, (255,), thickness=-1)

        elif roi_label_name and cont_label_name == roi_label_name:
            x_min, x_max = np.min(contour_coordinates[:, 0]), np.max(contour_coordinates[:, 0])
            y_min, y_max = np.min(contour_coordinates[:, 1]), np.max(contour_coordinates[:, 1])
            rois.append([x_min, y_min, x_max, y_max])
        else:
            print('unknown contour label name, skip...')

    # 当标注没有划定roi时，默认按照全部标注轮廓的最小外接矩作为唯一的roi ? TODO
    if not roi_label_name:
        print('TODO')

    # 按照roi进行区域切图
    print('roi num :', len(rois))
    for roi_index, roi in enumerate(rois):
        x_min, y_min, x_max, y_max = roi
        roi_img_array = img_ds[y_min:(y_max + 1), x_min:(x_max + 1)]
        roi_mask_array = mask_ds[y_min:(y_max + 1), x_min:(x_max + 1)]

        cut(roi_index, roi_img_array, roi_mask_array, save_dir, save_name, cut_size, overlap)
        print('---------------')


# 腐蚀标注区域的函数
def erode_shapes(shapes, erosion_factor, h, w):
    points = np.array(shapes['coordinates'][0], dtype=np.int32)

    # 创建空白图像
    img = np.zeros((h, w), dtype=np.uint8)
    # 在空白图像上绘制轮廓
    cv2.drawContours(img, [points], -1, (255), thickness=cv2.FILLED)
    # 腐蚀轮廓
    kernel = np.ones((erosion_factor, erosion_factor), np.uint8)
    img_eroded = cv2.erode(img, kernel, iterations=1)
    # 获取腐蚀后的轮廓
    contours_eroded, _ = cv2.findContours(img_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 更新原始标注区域
    shapes['coordinates'] = contours_eroded[0][:, 0, :].tolist()


if __name__ == '__main__':
    svs_dir = '../data/pre_data/svs'
    json_dir = '../data/pre_data/json'
    level = 1
    anno_label_name = '舌下腺腺泡'
    save_dir = '../data/pre_data/ds_json_reduce'
    cut_size = 640
    cut_stride = 400
    roi_label_name = '无属性'
    erosion_factor = 5
    ##############################################

    all_json_path = glob.glob(os.path.join(json_dir, '*.json'))
    print(len(all_json_path))
#     print(all_json_path)
    for json_path in all_json_path:
        print(json_path)
        name = json_path.split(os.sep)[-1].split('.')[0]
        svs_path = os.path.join(svs_dir, name + '.svs')

        process_one_slide(svs_path, json_path, level, anno_label_name, save_dir, cut_size, cut_stride, roi_label_name)

