# -*- encoding: utf-8 -*-
"""
@File    : find_roi_oophoron.py
@Time    : 2023/11/1
@Author  : 李亚波
@Usage   :
"""
import os
import cv2
import time
from load_model import LoadModel
from tool.read_img import ImgFunc, OpenSlideImg, TiffImg
from find_roi_oophoron import FindRoiOophoron
import numpy as np
from model_predict import Predict
from tool.read_img import array_to_STAI
from tool.st_json import WriteJson
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def pipline(slide_dir_path,debug_path,debug_path1,debug_path2):
    # yolo_model_1280_ds32 = r'E:\yolo7_pre\weights\qzq_best.pt'
    yolo_model_1280_ds1 = r'/home/lyb/data_yhr/03-yhr/External_project/yolov7-main/runs/train/exp_TILs_v2/weights/best.pt'
    # 1.加载模型
    load_model_start = time.time()
    model_path_list = [yolo_model_1280_ds1]

    model_list = LoadModel.kick_model(model_path_list)
    load_model_end = time.time()

    print('load model time:', load_model_end - load_model_start)
    predict = Predict(*model_list)
    debug_prefixes = []
    for debug_file in os.listdir(debug_path1):
        print(debug_file)
        debug_basename = os.path.basename(debug_file)
        save_file_prefix = debug_basename.split('_')[0]
        debug_prefixes.append(save_file_prefix)
    for debug_file in os.listdir(debug_path2):
        print(debug_file)
        debug_basename = os.path.basename(debug_file)
        save_file_prefix = debug_basename.rpartition('.')[0]
        debug_prefixes.append(save_file_prefix)
    print(debug_prefixes)
    for slide_file_path in get_all_file(slide_dir_path, '.svs'):
        print(slide_file_path)
        save_file = os.path.basename(slide_file_path).rpartition('.')[0]
        print(save_file)
        if save_file in debug_prefixes:
            print(f"break exits files: {save_file}")
            continue

#         os.makedirs(debug_path, exist_ok=True)

        img_read = ImgFunc()
        svs_read = OpenSlideImg(slide_file_path)
        svs_read1 = TiffImg(slide_file_path)
        img_ds1 = svs_read1.get_img_ds(1)
#         img_ds4 = svs_read.get_img_ds(4)
        img_ds32 = svs_read.get_img_ds(32)

        roi_xywhs = []
        h32, w32 = svs_read.get_img_ds_shape(32)
        # 2.初始化findroi
        find_roi = FindRoiOophoron(svs_read, img_ds32, predict)

        # 3.切片问题前处理
        roi_xywhs.append([int(0), int(0), int(w32), int(h32)])
        # roi_xywhs = find_roi.get_oop_tissue()
#         roi_xywhs = find_roi.pretreatment(cut_edge_rate=0.05)

        #获取roi外轮廓
        out_conts_ds32 = find_roi.get_out_conts(roi_xywhs)

        # 获取大卵泡目标coord_list
        coord_list_ds1, out_mask_ds32 = find_roi.get_roi_coord(out_conts_ds32, level_ds=1, size=640, step=320,rate=0.01)
        img_list_ds1 = img_read.get_patch_img_list(img_ds1, coord_list_ds1)

        # 获取大卵泡目标dec
        oop_coords_dict_ds1 = find_roi.detect_oop_big(img_list_ds1, coord_list_ds1)
        oop_coords_dict_ds1 = find_roi.filter_0_big(oop_coords_dict_ds1, out_mask_ds32)
        oop_coords_dict_ds1 = find_roi.cls_oop_1(oop_coords_dict_ds1)
        total_values_count = sum(len(value) for value in oop_coords_dict_ds1.values())

        #保存json
        oop_center_dict_ds1 = {}
#         svs_name_get_coors = slide_file_path.split('_')[-1].split('.')[0].split(',')
#         print(svs_name_get_coors)
        x_min ,y_min = 0,0
#         x_min ,y_min =svs_name_get_coors[0], svs_name_get_coors[1]
        print('x_min,y_min:',x_min,y_min)
        for label, coord_list in oop_coords_dict_ds1.items():
            # print('coord_list',coord_list)

            for coord in coord_list:
                coord = [[[coord[0][0], coord[0][1]]], [[coord[1][0], coord[1][1]]]]

                x1 = coord[0][0][0] + float(x_min)
                y1 = coord[0][0][1] + float(y_min)
                x2 = coord[1][0][0] + float(x_min)
                y2 = coord[1][0][1] + float(y_min)

                a1 = [[x1, y1]]
                b1 = [[x2, y1]]
                c1 = [[x2, y2]]
                d1 = [[x1, y2]]
                coord_1 = [a1, b1, c1, d1]
                if label not in oop_center_dict_ds1:
                    oop_center_dict_ds1[label] = [coord_1]
                else:
                    oop_center_dict_ds1[label].append(np.array(coord_1))

#         WriteJson(debug_path, slide_file_path, oop_center_dict_ds1).main()
        for xywh in roi_xywhs:  # 整个包含目标的大轮廓
            x, y, w, h = xywh
            x_center = x + w / 2  # 计算矩形框的中心点坐标
            y_center = y + h / 2
            # 画出中心点
            cv2.circle(img_ds1, (int(x_center*32), int(y_center*32)), 5, (255, 0, 0), -1)
            # 画出矩形框
            cv2.rectangle(img_ds1, tuple((int(x*32), int(y*32))), tuple((int((x + w)*32), int((y + h)*32))),
                          (255, 0, 0), 2)

        yuanxin=[]
        for label, coord_list in oop_coords_dict_ds1.items():
            label_color_dict = {'TILs': (255, 0, 0)}
            for coord in coord_list:
                # 计算矩形框的中心点坐标
                x_center = (coord[0][0] + coord[1][0]) / 2
                y_center = (coord[0][1] + coord[1][1]) / 2
                yuanxin.append((int(x_center), int(y_center)))
                # 画出中心点
#                 cv2.circle(img_ds1, (int(x_center), int(y_center)), 2, (0,0,255), 2)
                # 画出矩形框
                cv2.rectangle(img_ds1, tuple(coord[0]), tuple(coord[1]), label_color_dict[label], 1)
        yuanxin = np.array(yuanxin)
        npypath = os.path.join(debug_path, svs_read.slide_name) + ".npy"
        np.save(npypath, yuanxin)
        img_rgb_ds1 = cv2.cvtColor(img_ds1, cv2.COLOR_BGR2RGB)
        array_to_STAI(img_ds1, os.path.join(debug_path, svs_read.slide_name) + f'_{total_values_count}_new.svs')

#         #可视化
#         for xywh in roi_xywhs:  # 整个包含目标的大轮廓
#             x, y, w, h = xywh
#             x_max = x + w
#             y_max = y + h
#             cv2.rectangle(img_ds1, tuple((int(x*32), int(y*32))), tuple((int(x_max*32), int(y_max*32))),
#                           (255, 0, 0), 2)

#         for label, coord_list in oop_coords_dict_ds1.items():
#             label_color_dict = {'TLE': (255, 0, 0)}
#             for coord in coord_list:
#                 coord = np.array(coord)
#                 cv2.rectangle(img_ds1, tuple(coord[0]), tuple(coord[1]), label_color_dict[label], 3)

#         img_rgb_ds1 = cv2.cvtColor(img_ds1, cv2.COLOR_BGR2RGB)
#         array_to_STAI(img_rgb_ds1, os.path.join(debug_path, svs_read.slide_name) + '_new.svs')



def get_all_file(ori_path, postfix=''):
    """
    获取文件夹下所有文件
    :param ori_path: 文件地址
    :param postfix: 文件后缀
    :return: 文件地址列表
    """
    true_path_list = []
    for root, dirs, files in os.walk(ori_path, topdown=False):
        for name in files:
            if postfix:
                file_postfix = os.path.splitext(name)[-1]
                if file_postfix == postfix:
                    true_path_list.append(os.path.join(root, name))
            else:
                true_path_list.append(os.path.join(root, name))
    return true_path_list


if __name__ == '__main__':
    debug_path = '/home/lyb/data_yhr/03-yhr/External_project/Result_ALL_TGCA/260210_0.01'
    debug_path1 = '/home/lyb/data_yhr/03-yhr/External_project/Result_ALL_TGCA/'#TILs
    debug_path2 = '/home/lyb/data_yhr/03-yhr/External_project/Result_ALL_TGCA/'#TILs_add2
    os.makedirs(debug_path,exist_ok=True)
    svs_dir = '/home/lyb/data_yhr/03-yhr/External_project/Color-Transfer-between-Images/result_260209/'
#     svs_dir = '/home/lyb/data_yhr/03-yhr/External_project/yolo7_pre/svs_test/shf_try'
    
    pipline(svs_dir,debug_path,debug_path1,debug_path2)

