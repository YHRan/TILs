# -*- encoding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2021/12/9 14:41
@Author  : 高中需
@Usage   :
"""
import os
import numpy as np


class Coord_tool():
    def __init__(self):
        pass

    @classmethod
    def coord2xywh(cls, coord):
        w, h = coord[1] - coord[0]
        x, y = coord[0]
        return [x, y, w, h]

    @classmethod
    def xywh2coord(cls, x, y, w, h):
        return [[x, y], [x + w, y + h]]

    @classmethod
    def xywh2xyxy(cls, xywh):
        x, y, w, h = xywh
        return [x, y, x + w, y + h]

    @classmethod
    def xywh2angular_point(cls, x, y, w, h):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    @classmethod
    def get_all_file(cls, ori_path, postfix=''):
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

    @classmethod
    def transform_conts_level(cls, conts, conts_level_ds, roi_level_ds=1):
        """
        轮廓修改缩放比例

        :param conts: 轮廓
        :param conts_level_ds: 轮廓的缩放比例
        :param roi_level_ds: 目标缩放比例
        """
        return [np.array(cont * conts_level_ds / roi_level_ds, dtype='int') for cont in conts]
if __name__ == '__main__':
    import numpy as np
    a= np.array([1,2,3,4,5,6])
    b= [0,-1]
    print(a[b])