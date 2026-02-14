"""
@File    : find_roi_oophoron.py
@Time    : 2023/4/12 15:17
@Author  : 李亚波
@Usage   :
"""
from utils_v5.segment.general import process_mask
import torchvision
import os
import time
import torch
import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from random import randint
from skimage import filters
from skimage.morphology import disk

from models.v5_dataloader import LoadArrayImages,LoadImages
from model_predict import Predict
from find_roi import FindRoi
from tool.read_img import OpenSlideImg, ImgFunc, array_to_STAI, TiffImg
from tool.utils import Coord_tool
from models.utils import select_device, box_iou, scale_coords
from PIL import Image
Image.MAX_IMAGE_PIXELS = None



class FindRoiOophoron(FindRoi):

    def __init__(self, svs_read: OpenSlideImg or TiffImg, img_ds32, predict):
        super(FindRoiOophoron, self).__init__()
        self.predict = predict
        self.svs_read = svs_read
        self.img_ds32 = img_ds32
        self.img_read = ImgFunc()

    def get_oop_tissue(self):
        dataset_ds32 = LoadArrayImages([[[0, 0], [0, 0]]], [self.img_ds32], img_size=512) #1280
        boxes_ds32 = self.yolo_decode_ds32(dataset_ds32, model=self.predict.yolo_model_1280_ds32, conf_thres=0.3)
        roi_xywh = []
        for i, box in enumerate(boxes_ds32):
            # print(self.img_ds32.shape)
            # img = self.img_ds32[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            # cv2.imshow(str(i),img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            roi_xywh.append([int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])])
        #             roi_xywh.append([int(box[0]), int(box[1]), int((box[2] - box[0])*1.1), int((box[3] - box[1])*1.1)])
        return roi_xywh

    def pretreatment(self, cut_edge_rate):
        """
        用于处理切片边缘黑色区域

        Args:
            cut_edge_rate: 定义边缘区域比例

        Returns: [list]内部区域xywh

        """

        def x_in_scope(x, scope: list):
            return scope[0] <= x <= scope[1]

        def point_in_edge(point, img_shape, edge_rate):
            try:
                if edge_rate < 0 or edge_rate > 1:
                    raise
            except:
                print('error: edge_rate range in 0~1')
                return False
            h, w = img_shape
            temp = np.array([[0, edge_rate], [1 - edge_rate, 1]])
            for i, j in zip([w, h], point):
                for scope in i * temp:
                    if x_in_scope(j, scope):
                        return True
            return False

        gray_save = cv2.cvtColor(self.img_ds32, cv2.COLOR_BGR2GRAY)
        gray = (filters.roberts(gray_save) * 255).astype('uint8')
        ret3, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU二值化
        k_close = np.ones((3, 3), np.uint8)
        close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=3)
        _, mask_cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_xywhs = []
        for cnt in mask_cnts:
            if cv2.contourArea(cnt) > 10: #10000
                xywh = cv2.boundingRect(cnt)
                angular_point = Coord_tool.xywh2angular_point(*xywh)
                num = 0
                for point in angular_point:
                    if not point_in_edge(point, img_shape=self.img_ds32.shape[:2], edge_rate=cut_edge_rate):
                        num += 1
                if num == 4:
                    roi_xywhs.append(xywh)

        # for cnt in mask_cnts:
        #     # if cv2.contourArea(cnt) > 10000:
        #     xywh = cv2.boundingRect(cnt)
        #     angular_point = Coord_tool.xywh2angular_point(*xywh)
        #     # num = 0
        #     # for point in angular_point:
        #     #     if not point_in_edge(point, img_shape=self.img_ds32.shape[:2], edge_rate=cut_edge_rate):
        #     #         num += 1
        #     # if num == 4:
        #     roi_xywhs.append(xywh)
        return roi_xywhs



    def get_out_conts(self, roi_xywhs):
        """
        获取组织区域外轮廓

        Args:
            roi_xywhs: 所有目标区域xywh

        Returns: 组织外轮廓

        """
        mask_ds32 = np.zeros(self.img_ds32.shape[:2], dtype='uint8')
        for xywh in roi_xywhs:
            coord = Coord_tool.xywh2coord(*xywh)
            cv2.rectangle(mask_ds32, tuple(coord[0]), tuple(coord[1]), 255, -1)

        img_hsv_ds32 = cv2.cvtColor(self.img_ds32, cv2.COLOR_RGB2HSV)
        out_mask_ds32 = cv2.inRange(img_hsv_ds32, np.array([18, 9, 0]), np.array([180, 255, 255]))
        median = filters.median(out_mask_ds32, disk(4))
        mask_and = cv2.bitwise_and(median, mask_ds32)
        # median[mask_ds32 != 255] = 255
        _, cnts, _ = cv2.findContours(mask_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conts_ds32 = [cont for cont in cnts if cv2.contourArea(cont) > 10] #2000

        # out_mask_ds32 = np.zeros(self.img_ds32.shape[:2], dtype='uint8')
        # cv2.drawContours(out_mask_ds32,new_conts,-1,255,-1)
        #
        #
        # img_gray = cv2.cvtColor(self.img_ds32, cv2.COLOR_BGR2GRAY)
        # img_gray[mask_ds32 != 255] = 255
        # ret, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # median = filters.median(binary, disk(4))
        # _, cnts, _ = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # conts_ds32 = [cont for cont in cnts if cv2.contourArea(cont) > 2000]
        return conts_ds32



    def get_roi_coord(self, out_conts_ds32, level_ds, size, step, rate=0.01):
        """
        获取目标区域切割小图坐标。

        Args:
            out_conts_ds32:
            level_ds:
            size:
            step:
            rate:
        Returns:

        """
        h, w = self.svs_read.get_img_ds_shape(32)
        out_mask_ds32 = np.zeros((h, w), dtype='uint8')
        cv2.drawContours(out_mask_ds32, out_conts_ds32, -1, 255, -1)

        # cv2.imwrite(str(round(time.time(),4))+'test.png',out_mask_ds32)
        coord_list_ds4 = self.get_patch_list(out_mask_ds32, 32 / level_ds, size=size, step=step, rate=rate)
        return coord_list_ds4, out_mask_ds32

    def yolo_decode_ds32(self, dataset, model, conf_thres):
        all_coord = []
        device = select_device(0)
        for coord, im, im0s in dataset:
            h, w, c = im0s.shape
            pred = self.predict.yolo_predict(im, model=model, conf_thres=conf_thres, device=device)
            for i, det in enumerate(pred):  # per image
                if len(det):
                    det[:, :4] = scale_coords(im.shape[1:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        cls_list = int(cls.tolist())
#                         print('cls_list',cls_list)
                        if cls_list == 13:
                            xyxy = torch.tensor(xyxy).view(-1, 4)
                            xyxy = np.array(xyxy).squeeze()

                            if xyxy[0] < 0.02 * w:
                                xyxy[0] = 0
                            if xyxy[1] < 0.02 * h:
                                xyxy[1] = 0
                            if xyxy[2] > 0.98 * w:
                                xyxy[2] = w
                            if xyxy[3] > 0.98 * h:
                                xyxy[3] = h

                            temp_ = np.array(xyxy) + np.array([*coord[0], *coord[0]])
                            temp_ = list(temp_)
                            temp_.append(float(conf.cpu().numpy()))
                            temp_.append(int(cls.cpu().numpy()))
                            all_coord.append(np.array(temp_))
        return all_coord

    def yolo_decode(self, dataset, model, conf_thres):
        all_coord = []
        device = select_device(0)
#         for coord, im, im0s in dataset:
        for i, (coord, im, im0s) in enumerate(tqdm(dataset, desc='Processing dataset')):
            h, w, c = im0s.shape
            pred = self.predict.yolo_predict(im, model=model, conf_thres=conf_thres, device=device)
            for i, det in enumerate(pred):  # per image
#             for i, det in tqdm(enumerate(pred), total=len(pred)):
                if len(det):
                    det[:, :4] = scale_coords(im.shape[1:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = torch.tensor(xyxy).view(-1, 4)
                        xyxy = np.array(xyxy).squeeze()

                        if xyxy[0] < 0.02 * w:
                            xyxy[0] = 0
                        if xyxy[1] < 0.02 * h:
                            xyxy[1] = 0
                        if xyxy[2] > 0.98 * w:
                            xyxy[2] = w
                        if xyxy[3] > 0.98 * h:
                            xyxy[3] = h

                        temp_ = np.array(xyxy) + np.array([*coord[0], *coord[0]])
                        temp_ = list(temp_)
                        temp_.append(float(conf.cpu().numpy()))
                        temp_.append(int(cls.cpu().numpy()))
                        all_coord.append(np.array(temp_))

        return all_coord



    def detect_oop_big(self, img_list_ds4, coord_list_ds4):
        dataset_ds4 = LoadArrayImages(coord_list_ds4, img_list_ds4)
        boxes_ds4 = self.yolo_decode(dataset_ds4, model=self.predict.yolo_model_1280_ds1, conf_thres=0.01)# 0.01 0.005
#         merge_ds4 = MergeDetectCoords2(boxes_ds4, coord_list_ds4, interstice=8).main()  # 处理拼接缝问题。  细胞注释掉
        print("boxes_ds4 have done!")
        print("num:",len(boxes_ds4))
        nms_dict_ds4 = _my_iou(boxes_ds4, {0: 'TILs'}, 0.3)  # 拼接之后，过滤掉错误目标0.3
        # 使用生成器表达式计算所有值中元素的总数
        total_values_count = sum(len(value) for value in nms_dict_ds4.values())
        print(f"total {total_values_count}")
        print("nms_dict_ds4 have done!")
        # dict_box_list = []
        # for i in nms_dict_ds4.values():
        #     for j in i:
        #         dict_box_list.append(j.tolist())
        #
        # dict_area_list = []
        # for i in dict_box_list:
        #     dict_area_list.append((i[2] - i[0]) * (i[3] - i[1]))
        #
        # list_area_box_zip = list(zip(dict_area_list, dict_box_list))
        # area_box_zip_sort = sorted(list_area_box_zip, key=lambda x: -x[0])
        # area_box_result = zip(*area_box_zip_sort)
        # new_area_all, new_box_all = [list(x) for x in area_box_result]
        #
        # new_box_all_centre = []
        # new_box_all_1 = []
        # for i in range(len(new_box_all)):
        #     x_centre = int(((new_box_all[i][2] - new_box_all[i][0]) / 2) + new_box_all[i][0])
        #     y_centre = int(((new_box_all[i][3] - new_box_all[i][1]) / 2) + new_box_all[i][1])
        #     new_box_all_centre.append([x_centre, y_centre])
        #     new_box_all_1.append([int(new_box_all[i][0] + ((new_box_all[i][2] - new_box_all[i][0]) * 0.05)),
        #                           int(new_box_all[i][1] + ((new_box_all[i][3] - new_box_all[i][1]) * 0.05)),
        #                           int(new_box_all[i][2] - ((new_box_all[i][2] - new_box_all[i][0]) * 0.05)),
        #                           int(new_box_all[i][3] - ((new_box_all[i][3] - new_box_all[i][1]) * 0.05))])
        #
        # del_new_box_all = []
        # for i in range(len(new_box_all_1) - 1):
        #     for j in range(i + 1, len(new_box_all_1)):
        #         if new_box_all_1[i][0] <= new_box_all_centre[j][0] <= new_box_all_1[i][2] and new_box_all_1[i][1] <= \
        #                 new_box_all_centre[j][1] <= new_box_all_1[i][3]:
        #             del_new_box_all.append(new_box_all[j])
        #
        # # 需要删除的数据
        # del_new_box_all_1 = []
        # # 去重
        # for li in del_new_box_all:
        #     if li not in del_new_box_all_1:
        #         del_new_box_all_1.append(li)
        #
        # # 获取下标
        # idx_list = []
        # for i in del_new_box_all_1:
        #     a = new_box_all.index(i)
        #     idx_list.append(a)
        #
        # new_box_all_idx = []
        # for i in range(len(new_box_all)):
        #     new_box_all_idx.append(i)
        #
        # for j in idx_list:
        #     for k in new_box_all_idx:
        #         if j == k:
        #             new_box_all_idx.remove(k)
        #
        # new_box_all_list = []
        # dict_ds2_os = {}
        # for i in new_box_all_idx:
        #     new_box_all_list.append(np.array(new_box_all[i]) * 2)
        #     dict_ds2_os['os'] = new_box_all_list

        return nms_dict_ds4


    def filter_0_big(self, oop_coords_dict_ds4, out_mask_ds32):
        new_coords_dict = {'TILs': []}
        for label, coord_list in oop_coords_dict_ds4.items():
            for coord_ in coord_list:
                coord = [np.array([coord_[0], coord_[1]], dtype='int32'),
                         np.array([coord_[2], coord_[3]], dtype='int32')]
                center = np.array((coord[1] + coord[0]) / 2, dtype='int32')
                if out_mask_ds32[int(center[1] / 32) - 1, int(center[0] / 32) - 1] == 255:
                    new_coords_dict[label].append(coord_)

        return new_coords_dict

    def cls_oop_1(self, oop_conts_dict_ds1):
        coord_dict = {}
        for label, coord_list in oop_conts_dict_ds1.items():
            print('*' * 10, label, '*' * 10)
            for coord in coord_list:
                coord = [np.array([coord[0], coord[1]], dtype='int32'), np.array([coord[2], coord[3]], dtype='int32')]
                result = 'TILs'
                if result not in coord_dict.keys():
                    coord_dict[result] = []
                coord_dict[result].append(coord)
        return coord_dict

class MergeDetectCoords2():
    def __init__(self, coord_list, patch_coord_list, interstice):
        """
        拼接位于拼接缝处的回归框。

        Args:
            coord_list: 所有回归框
            patch_coord_list: patch图的框
        """
        self.coord_list = coord_list
        self.patch_coord_list = np.array(patch_coord_list)
        self.interstice = interstice
        # self.img_rgb_ds2 = img_rgb_ds2

    def distance(self, line):
        return np.linalg.norm(line[0] - line[1])

    # def collineation(self, line1, line2, direction='x', interstice=8):
    #     """
    #     获取线段的共线部分。
    #
    #     Args:
    #         line1: 线段1
    #         line2: 线段2
    #         direction: 轴向
    #         interstice: 非共线容忍宽度
    #
    #     Returns:
    #
    #     """
    #
    #     temp = line1 - line2
    #     if direction == 'x':
    #         if temp[1] > interstice or temp[1] < -interstice:
    #             return []
    #
    #         point_list_1 = line1[0:4:2].ravel()
    #         point_list_2 = line2[0:4:2].ravel()
    #     else:
    #         if temp[0] > interstice or temp[0] < -interstice:
    #             return []
    #         point_list_1 = line1[1:4:2].ravel()
    #         point_list_2 = line2[1:4:2].ravel()
    #
    #     code_str = ''
    #     for x1 in point_list_1:
    #         for x2 in point_list_2:
    #             if x1 <= x2:
    #                 code_str += '1'
    #             else:
    #                 code_str += '0'
    #
    #     if code_str == '1111' or code_str == '0000':
    #         return []
    #     elif code_str == '1101':
    #         return [line2[0], line2[1], line1[2], line1[3]]
    #     elif code_str == '1100':
    #         return line2
    #     elif code_str == '0100':
    #         return [line1[0], line1[1], line2[2], line2[3]]
    #     elif code_str == '0101':
    #         return line1
    #     else:
    #         print('边界拼接错误')
    #         return []
    def straight_line_iou(self, line1, line2, tolerance):
        def axis(line, tolerance=3):
            if -tolerance < line[2] - line[0] < tolerance:
                return ['y', line[0], [line[1], line[3]]]

            elif -tolerance < line[3] - line[1] < tolerance:
                return ['x', line[1], [line[0], line[2]]]

            else:
                return []

        axis1 = axis(line1, tolerance)
        axis2 = axis(line2, tolerance)

        if axis1[0] == axis2[0] and -tolerance < axis1[1] - axis2[1] < tolerance:

            length1 = axis1[2][1] - axis1[2][0]
            length2 = axis2[2][1] - axis2[2][0]
            inter = max(min(axis1[2][1], axis2[2][1]) - max(axis1[2][0], axis2[2][0]), 0)  # 限制范围在0以上
            return inter, length1, length2
            # return inter / (length1 + length2 - inter)
        else:
            return 0, 0, 0

    def judge_merge(self, line1, line2, score=0.3, interstice=10):
        """
        判断两条线段是否合并。

        Args:
            line1: 线段1
            line2: 线段2
            direction: 轴向
            score: 共线长度/短线长度

        Returns:
            是否合并。
        """
        # collineation_line = self.collineation(line1, line2, direction=direction, interstice=interstice)
        inter, length1, length2 = self.straight_line_iou(line1, line2, tolerance=interstice)
        if inter != 0:
            short_line = min(length1, length2)
            long_line = max(length1, length2)
            short_iou = inter / short_line
            long_iou = inter / long_line
            # print(line1,line2)
            # print(short_iou,long_iou)
            if short_iou > score and long_iou > 0.3:
                return True
            else:
                return False

    # def judge_merge_2(self, line1, line2, score=0.3, tolerance=10):
    #     """
    #     判断两条线段是否合并。
    #
    #     Args:
    #         line1: 线段1
    #         line2: 线段2
    #         score: iou
    #
    #     Returns:
    #         是否合并。
    #     """
    #
    #     def axis(line, tolerance=3):
    #         if -tolerance < line[2] - line[0] < tolerance:
    #             return ['y', [line[1], line[3]]]
    #         elif -tolerance < line[3] - line[1] < tolerance:
    #             return ['x', [line[0], line[2]]]
    #         else:
    #             return []
    #
    #     axis1 = axis(line1, tolerance)
    #     axis2 = axis(line2, tolerance)
    #     if axis1[0] == axis2[0]:
    #         length1 = axis1[1][1] - axis1[1][0]
    #         length2 = axis2[1][1] - axis2[1][0]
    #         inter = max(min(axis1[1][1], axis2[1][1]) - max(axis1[1][0], axis2[1][0]), 0)  # 限制范围在0以上
    #         iou = inter / (length1 + length2 - inter)
    #         if iou > score:
    #             return True
    #         else:
    #             return False
    #     else:
    #         return False

    def filter_coord(self, coord_list, patch_coord_list):
        """
        筛选位于拼接缝处的框。

        Returns: need_fuse_coord_list(list)需要拼接的框，single_coord_list(list)独立的框

        """
        patch_coord_list_ = patch_coord_list.reshape(-1, 4)
        y_list = set(patch_coord_list_[:, 1:4:2].ravel())
        x_list = set(patch_coord_list_[:, 0:4:2].ravel())

        need_fuse_coord_list = []
        single_coord_list = []
        for coord in coord_list:
            score = 0

            for x in coord[0:4:2].ravel():
                for patch_x in x_list:
                    if patch_x - self.interstice < x < patch_x + self.interstice:
                        score += 1

            for y in coord[1:4:2].ravel():
                for patch_y in y_list:
                    if patch_y - self.interstice < y < patch_y + self.interstice:
                        score += 1

            if score > 0:
                need_fuse_coord_list.append(coord)
            else:
                single_coord_list.append(coord)

        return need_fuse_coord_list, single_coord_list

    def get_lines(self, coord) -> object:
        """
        返回coord格式框的四条边。

        Args:
            coord: 框坐标数组：[[x_min,y_min],[x_max,y_max]]

        Returns:
            top(array)上边；bottom(array)下边；left(array)左边；right(array)右边。
        """

        top = [coord[0], coord[1], coord[2], coord[1]]
        bottom = [coord[0], coord[3], coord[2], coord[3]]
        left = [coord[0], coord[1], coord[0], coord[3]]
        right = [coord[2], coord[1], coord[2], coord[3]]
        return np.array(top), np.array(bottom), np.array(left), np.array(right)

    def fuse_coord(self, coord_1, coord_2):
        """
        合并框体。

        Args:
            coord_1: 框1
            coord_2: 框2

        Returns:合并后的框。

        """
        temp_x = list(coord_1[0:4:2].ravel()) + list(coord_2[0:4:2].ravel())
        temp_y = list(coord_1[1:4:2].ravel()) + list(coord_2[1:4:2].ravel())
        conf = coord_1[-2]
        cls = coord_1[-1]
        if coord_1[-2] < coord_2[-2]:
            conf = coord_2[-2]
            cls = coord_2[-1]

        if coord_1[-2] == coord_2[-2]:
            if cls == 1 or coord_2[-1] == 1:
                cls = 1

        new_coord = [min(temp_x), min(temp_y), max(temp_x), max(temp_y), conf, int(cls)]
        return np.array(new_coord)

    def coord_in_patch_margin(self, x_list, y_list, coord):
        size = int(self.interstice) / 2
        x_line_dict = {0: 'line_left', 1: 'line_right'}
        y_line_dict = {0: 'line_top', 1: 'line_bottom'}
        fuse_line = []
        for i, x in enumerate(coord[0:4:2].ravel()):
            for patch_x in x_list:
                if patch_x - size < x < patch_x + size:
                    fuse_line.append(x_line_dict[i])

        for i, y in enumerate(coord[1:4:2].ravel()):
            for patch_y in y_list:
                if patch_y - size < y < patch_y + size:
                    fuse_line.append(y_line_dict[i])

        return fuse_line

    def get_four_quadrant(self, point, size=10):
        point = np.array(point, dtype=np.int32)
        return np.array([point + [-size, size], point + [size, size], point + [size, -size], point + [-size, -size]])

    def point_in_coord(self, point, coord):
        if coord[0] < point[0] < coord[2] and coord[1] < point[1] < coord[3]:
            return True
        return False

    def fuse_vertex_boxes(self, boxes):
        boxes = np.array(boxes)
        x_min = np.min(boxes[:, 0:4:2].ravel())
        x_max = np.max(boxes[:, 0:4:2].ravel())
        y_min = np.min(boxes[:, 1:4:2].ravel())
        y_max = np.max(boxes[:, 1:4:2].ravel())

        conf = boxes[0][-2]
        cls = boxes[0][-1]
        for box in boxes[1:]:
            if box[-2] > conf:
                conf = box[-2]
                cls = box[-1]
        return np.array([x_min, y_min, x_max, y_max, conf, cls])

    # def get_vertex_points(self,patch_coord_list):
    #     for coord in patch_coord_list:

    def get_vertex_object(self, boxes, patch_coord_list):
        # img_rgb_ds2 = self.img_rgb_ds2.copy()
        # for coord in patch_coord_list:
        #     cv2.rectangle(img_rgb_ds2, tuple(coord[0].astype(np.int32)), tuple(coord[1].astype(np.int32)), (0, 0, 0), 2)
        #
        # for box in boxes:
        #     cv2.rectangle(img_rgb_ds2, tuple([int(box[0]), int(box[1])]), tuple([int(box[2]), int(box[3])]),
        #                   (255, 255, 0), 2)

        merget_index_list = []
        merge_list = []
        patch_coord_list = np.array(patch_coord_list)
        for vertex_point in patch_coord_list.reshape(-1, 2):  # 贴片的左上、右下角点
            num = 0
            merge_index = []
            for point in self.get_four_quadrant(vertex_point, size=20):  # 角点的四邻域点
                # cv2.circle(img_rgb_ds2, tuple(point), radius=3, color=(0, 0, 255), thickness=-1)
                temp = []
                for i, coord in enumerate(boxes):  # 所有预测框
                    if i not in merget_index_list and self.point_in_coord(point, coord):
                        # if self.point_in_coord(point, coord):
                        num += 1
                        temp.append(i)

                if len(temp) > 1:
                    merge_index.append(min(temp))
                else:
                    merge_index += temp

            # if num ==3:

            if num >= 4:
                # print(num)
                # print(merge_index)
                merget_index_list += merge_index
                merge_box = self.fuse_vertex_boxes(np.array(boxes)[merge_index])
                merge_list.append(merge_box)

        single_list = [boxes[i] for i in range(0, len(boxes)) if i not in merget_index_list]

        # for box in np.array(boxes)[merget_index_list]:
        #     cv2.rectangle(img_rgb_ds2, tuple([int(box[0]), int(box[1])]), tuple([int(box[2]), int(box[3])]),
        #                   (255, 0, 255), 1)
        #
        # array_to_STAI(img_rgb_ds2, os.path.join(r'D:\data\oophoron\debug\debug_1', 'debug') + '.svs')

        return merge_list, single_list

    def main(self):
        need_merge_coord_list, single_coord_list = self.filter_coord(self.coord_list, self.patch_coord_list)
        merge_coord_list, all_red_box = self.get_vertex_object(need_merge_coord_list, self.patch_coord_list)
        all_red_box += merge_coord_list

        y_list = set(self.patch_coord_list[:, 1:4:2].ravel())
        x_list = set(self.patch_coord_list[:, 0:4:2].ravel())
        match_dict = {'line_left': 'line_right_2',
                      'line_right': 'line_left_2',
                      'line_top': 'line_bottom_2',
                      'line_bottom': 'line_top_2'}  # 允许拼接的边

        hold_box = []
        while all_red_box:
            double_check = False
            coord1 = all_red_box.pop(0)
            times_check = len(all_red_box)
            while times_check != 0:
                coord2 = all_red_box.pop(0)

                line_top, line_bottom, line_left, line_right = self.get_lines(coord1)
                line_top_2, line_bottom_2, line_left_2, line_right_2 = self.get_lines(coord2)
                fuse_line = self.coord_in_patch_margin(x_list, y_list, coord1)
                flag = 0
                for line_1 in fuse_line:
                    line_2 = match_dict[line_1]
                    if line_1 in ['line_left', 'line_right']:
                        direction = 'y'
                    else:
                        direction = 'x'

                    if self.judge_merge(eval(line_1), eval(line_2), score=0.8, interstice=self.interstice):  # 判断是否拼接
                        flag += 1

                # flag, merged_box = A(coord1, coord2)
                if flag > 0:  # merge
                    merged_box = self.fuse_coord(coord1, coord2)
                    all_red_box.insert(0, merged_box)
                    double_check = True
                    break
                else:  # unmerge
                    all_red_box.append(coord2)
                    times_check -= 1

            if double_check == False:
                hold_box.append(coord1)

        result = hold_box + single_coord_list
        return result


def _my_iou_pre(x, label_dict, iou_thres=0.3):
    x = np.array(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x).float().to(device)  # 确保数据是float类型，并且移动到GPU
    
#     device = 'cpu'
#     x = torch.from_numpy(x).to(device)
    boxes, scores = x[:, :4], x[:, 4]
    ious = box_iou(boxes, boxes)
    ious = torch.triu(ious, diagonal=1)  # 取上三角矩阵
    index = torch.where(ious > iou_thres)
    iou_list = ious[index].numpy()
    del_list = []
    for i, iou in enumerate(iou_list):

        if x[index[0][i]][-2] > x[index[1][i]][-2]:
            del_list.append(index[1][i].numpy())
        else:
            del_list.append(index[0][i].numpy())

    all_data_dict = {}
    for i, box in enumerate(x):
        if i in (del_list):
            continue
        label = label_dict[int(box[-1])]
        if label not in all_data_dict:
            all_data_dict[label] = []
        all_data_dict[label].append(np.array([int(box[0]), int(box[1]), int(box[2]), int(box[3])]))

    return all_data_dict


def _my_iou(x, label_dict, iou_thres=0.3, chunk_size=1000):
    x = np.array(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x).float().to(device)  # 确保数据是float类型，并且移动到GPU

    boxes, scores = x[:, :4], x[:, 4]
    
    # 处理过程中将数据分块
    del_list = set()
    for i in range(0, len(boxes), chunk_size):
        end = min(i + chunk_size, len(boxes))
        boxes_chunk = boxes[i:end]
        scores_chunk = scores[i:end]
        
        ious = box_iou(boxes_chunk, boxes_chunk)
        if ious is None:
            raise ValueError("ious is None, check the input boxes.")
        ious = torch.triu(ious, diagonal=1)  # 取上三角矩阵
        index = torch.where(ious > iou_thres)
        
        for idx in range(len(index[0])):
            if scores_chunk[index[0][idx]] > scores_chunk[index[1][idx]]:
                del_list.add(i + index[1][idx].item())
            else:
                del_list.add(i + index[0][idx].item())
    
    all_data_dict = {}
    for i, box in enumerate(x):
        if i in del_list:
            continue
        label = label_dict[int(box[-1])]
        if label not in all_data_dict:
            all_data_dict[label] = []
        all_data_dict[label].append(np.array([int(box[0]), int(box[1]), int(box[2]), int(box[3])]))
    
    # Clear cache after processing to free up memory
    torch.cuda.empty_cache()
    
    return all_data_dict

# Helper function to calculate IOU (Intersection over Union)
def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

if __name__ == '__main__':
    pass
