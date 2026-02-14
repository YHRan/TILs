# -*- encoding: utf-8 -*-
"""
@File    : find_roi.py
@Time    : 2021/12/2 14:47
@Author  : 高中需
@Usage   :
"""
import os
import cv2
import numpy as np
from math import *
from PIL import Image
import itertools


class FindRoi():
    def __init__(self):
        pass

    def find_foreground(self, img, low, high):
        rgb_img = img.copy()
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        lower_color = np.array(low)
        upper_color = np.array(high)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        return mask, hsv

    def get_white_mask(self, img_ds):
        lab = cv2.cvtColor(img_ds, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1]
        mask = a.copy()
        mask = cv2.inRange(mask, 0, 140)
        return mask

    def get_blood_mask(self, img_ds):
        bgr = img_ds[:, :, :3]
        mask, _ = self.find_foreground(bgr, low=[125, 0, 0], high=[136, 255, 240])
        return np.array(mask)

    def gauss_smooth_conts(self, small_cnts, template=[-8, -5, -3, 0, 3, 5, 8], sigema=10, step=1):
        """
        对轮廓进行高斯平滑
        :param small_cnts: findconts输出的轮廓
        :param template: 高斯核取值
        :param sigema: 标准差
        :param step: 轮廓降采样步长
        :return: 平滑后的轮廓
        """

        def _gauss(small_cnts, template, sigema=10, step=1):
            # 高斯核
            new_template = []
            for x in template:
                _x = 1 / ((2 * pi) ** 0.5 * sigema) * exp(-(x ** 2 / 2 / (sigema ** 2)))
                new_template.append(_x)
            template = np.array(new_template) * (1 / sum(new_template))

            # 补位
            small_cnts = small_cnts[1 - len(template) // 2:] + small_cnts
            small_cnts = small_cnts + small_cnts[:len(template) // 2]
            new_cnts_ = []
            for i in range(len(template), len(small_cnts) + 1, step):
                t = np.array(small_cnts[i - len(template):i])
                a = np.multiply(t, template).sum()
                new_cnts_.append(int(a))
            return new_cnts_

        if len(template) % 2 != 1:
            print('need odd-length')
        new_cnts = []
        for cnt in small_cnts:
            cnt_temp = cnt.reshape(-1, 2)
            cnt_x = cnt_temp[:, 0]
            cnt_y = cnt_temp[:, 1]
            new_x = _gauss(list(cnt_x), template, sigema, step)
            new_y = _gauss(list(cnt_y), template, sigema, step)
            temp_cnts = []
            for i in range(len(new_x)):
                temp_cnts.append([[new_x[i], new_y[i]]])
            new_cnts.append(np.array(temp_cnts))
        return new_cnts

    def transform_conts_level(self, conts, level_ds, roi_level_ds=1):
        level_ds = level_ds
        new_conts = []
        for cont in conts:
            new_conts.append((cont * level_ds / roi_level_ds).astype('int'))
        return new_conts

    def pad_img(self, img, factor=256):
        '''
        make sure the single pic :width//factor=0;height//factor=0
        :param img: cv2 returned array
        :param factor:
        :return:
        '''
        h, w = img.shape[0], img.shape[1]
        mod_w = w // factor
        res_w = w % factor
        mod_h = h // factor
        res_h = h % factor
        if res_w == 0:
            _32x_w = (mod_w) * factor
        else:
            _32x_w = (mod_w + 1) * factor
        if res_h == 0:
            _32x_h = (mod_h) * factor
        else:
            _32x_h = (mod_h + 1) * factor

        mask = np.zeros((_32x_h, _32x_w, 3)).astype(np.uint8)
        mask_ = Image.fromarray(mask)
        img_ = Image.fromarray(img)
        mask_.paste(img_, (0, 0))
        mask_ = np.array(mask_)
        return mask_

    def get_patch_list(self, out_mask, level_ds, size=640, step=640, rate=-1.):
        """

        :param out_mask:
        :param level_ds:
        :param size:
        :param step:
        :param rate:
        :return:
        """

        def _check_point(y, x, factor, neighbour):
            """
            生成检测点

            :param y: 左上角点y
            :param x: 左上角点x
            :param factor: 内缩比例
            :param neighbour: 图片大小
            :return: 检测点坐标[y,x]
            """
            factor_list = np.array([factor, 1 - factor]) * neighbour
            yield [y, x] + np.array(neighbour / 2, dtype='int')
            for i in itertools.product(factor_list, repeat=2):
                yield [y, x] + np.array(i, dtype='int')

        def check_img(y, x, img_mask, factor, neighbour):
            for y, x in _check_point(y, x, factor, neighbour):
                if img_mask[y, x] != 255:
                    return False
            return True

        neighbourhood = int(size / level_ds)
        step_size = int(step / level_ds)
        h, w = out_mask.shape[:2]

        coord_list = []
        for x in range(0, w, step_size):
            # if x + neighbourhood > w:
            #     break
            for y in range(0, h, step_size):
                # if y + neighbourhood > h:
                #     break
                if rate == -1:  # 检测四角点内缩和中心点
                    if check_img(y, x, out_mask, 0.16, neighbourhood):
                        coord_list.append([[int(x * level_ds), int(y * level_ds)],
                                           [int(x * level_ds + size), int(y * level_ds + size)]])
                elif rate == 0:  # 全切
                    coord_list.append([[int(x * level_ds), int(y * level_ds)],
                                       [int(x * level_ds + size), int(y * level_ds + size)]])
                else:  # 检测目标的占比
                    area = cv2.countNonZero(out_mask[y:y + neighbourhood, x:x + neighbourhood])
                    rate_ = area / (neighbourhood * neighbourhood)
                    if rate_ > rate:
                        coord_list.append([[int(x * level_ds), int(y * level_ds)],
                                           [int(x * level_ds + size), int(y * level_ds + size)]])
        return coord_list

    def fuse_detect_coords(self, coords_dict, interstice=2):
        def x_collineation(line_1, line_2, interstice=2):
            temp = line_1 - line_2
            if temp[0][1] > interstice or temp[0][1] < -interstice:
                return []

            code_str = ''
            for x1 in line_1[:, 0]:
                for x2 in line_2[:, 0]:
                    if x1 <= x2:
                        code_str += '1'
                    else:
                        code_str += '0'

            if code_str == '1111' or code_str == '0000':
                return []
            elif code_str == '1101':
                return [line_2[0], line_1[1]]
            elif code_str == '1100':
                return line_2
            elif code_str == '0100':
                return [line_1[0], line_2[1]]
            elif code_str == '0101':
                return line_1
            else:
                print('边界拼接错误')
                return []

        def y_collineation(line_1, line_2, interstice=2):
            temp = line_1 - line_2
            if temp[0][0] > interstice or temp[0][0] < -interstice:
                return []

            code_str = ''
            for y1 in line_1[:, 1]:
                for y2 in line_2[:, 1]:
                    if y1 <= y2:
                        code_str += '1'
                    else:
                        code_str += '0'

            if code_str == '1111' or code_str == '0000':
                return []
            elif code_str == '1101':
                return [line_2[0], line_1[1]]
            elif code_str == '1100':
                return line_2
            elif code_str == '0100':
                return [line_1[0], line_2[1]]
            elif code_str == '0101':
                return line_1
            else:
                print('边界拼接错误')
                return []

        def distance(line):
            return np.linalg.norm(line[0] - line[1])

        def get_lines(coord):
            line_top = [coord[0], [coord[1][0], coord[0][1]]]
            line_bottom = [[coord[0][0], coord[1][1]], coord[1]]
            line_left = [coord[0], [coord[0][0], coord[1][1]]]
            line_right = [[coord[1][0], coord[0][1]], coord[1]]
            return np.array(line_top), np.array(line_bottom), np.array(line_left), np.array(line_right)

        coord_ds16 = coords_dict['oop']
        coord_list = list(coord_ds16)
        new_coord_list = []
        i_list = [i for i in range(len(coord_list))]

        for i in itertools.combinations(range(len(coord_list)), 2):
            coord = coord_list[i[0]]
            coord_2 = coord_list[i[1]]
            line_top, line_bottom, line_left, line_right = get_lines(coord)
            line_top_2, line_bottom_2, line_left_2, line_right_2 = get_lines(coord_2)
            for line_y_1, line_y_2 in [[line_left, line_left_2], [line_left, line_right_2], [line_right, line_left_2],
                                       [line_right, line_right_2]]:
                collineation_line = y_collineation(line_y_1, line_y_2, interstice)
                if len(collineation_line) != 0:
                    iou = distance(collineation_line) / (
                            distance(line_y_1) + distance(line_y_2) - distance(collineation_line))
                    if iou > 0.3:
                        temp_x = list(coord[:, 0]) + list(coord_2[:, 0])
                        temp_y = list(coord[:, 1]) + list(coord_2[:, 1])
                        new_coord = [[min(temp_x), min(temp_y)], [max(temp_x), max(temp_y)]]
                        new_coord_list.append(new_coord)
                        try:
                            i_list.remove(i[0])
                            i_list.remove(i[1])
                        except:
                            pass

            for line_x_1, line_x_2 in [[line_top, line_top_2], [line_top, line_bottom_2], [line_bottom, line_top_2],
                                       [line_bottom, line_bottom_2]]:
                collineation_line = x_collineation(line_x_1, line_x_2, interstice)
                if len(collineation_line) != 0:
                    iou = distance(collineation_line) / (
                            distance(line_x_1) + distance(line_x_2) - distance(collineation_line))
                    if iou > 0.3:
                        temp_x = list(coord[:, 0]) + list(coord_2[:, 0])
                        temp_y = list(coord[:, 1]) + list(coord_2[:, 1])
                        new_coord = [[min(temp_x), min(temp_y)], [max(temp_x), max(temp_y)]]
                        new_coord_list.append(new_coord)
                        try:
                            i_list.remove(i[0])
                            i_list.remove(i[1])
                        except:
                            pass

        for i in i_list:
            new_coord_list.append(coord_list[i])

        temp_ = [np.array(temp) for temp in new_coord_list]
        return {'oop': temp_}

    def nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        # x1、y1、x2、y2、以及score赋值
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        # 每一个检测框的面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 按照score置信度降序排序
        order = scores.argsort()[::-1]

        keep = []  # 保留的结果框集合
        while order.size > 0:
            i = order[0]
            keep.append(i)  # 保留该类剩余box中得分最高的一个
            # 得到相交区域,左上及右下
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算相交的面积,不重叠时面积为0
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 计算重叠面积占较小面积的比例。用于删除大框内的小框。
            ovr_ = inter / np.minimum(areas[i], areas[order[1:]])
            ovr__ = (ovr <= thresh) * (ovr_ <= 0.8)
            inds = np.where(ovr__)[0]

            # 保留IoU小于阈值的box
            # inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

        # temp = np.array(dets, dtype='int')[:, :4]
        # temp = temp.reshape(-1, 2, 2)
        result = [dets[i] for i in keep]
        return result
