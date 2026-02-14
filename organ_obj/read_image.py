# -*- encoding: utf-8 -*-
"""
@File    : read_img.py
@Time    : 2021/10/26 17:44
@Author  : 高中需
@Usage   : ImgFunc -> img读图，OpenSlideImg -> openlide读图，TiffImg -> tiffimg读图，mp_get_coord_img_list -> 多进程读图
"""
import os
import cv2
import math
import tifffile
import openslide
import numpy as np
from multiprocessing import Pool
from random import randint


class ImgFunc():

    def get_patch_img(self, img, coord):
        '''

        获取坐标内的小图



        :param img: img

        :param coord: [[xmin,ymin],[xmax,ymax]]

        :return: patch_img

        '''

        patch_img = img[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]

        return patch_img

    def get_patch_img_list(self, img, coord_list):
        """

        读取一批图片



        :param img: 原始图

        :param coord_list: [[[xmin,ymin],[xmax,ymax]],

                            [[xmin,ymin],[xmax,ymax]],...]

        :return:

        """

        patch_img_list = []

        for coord in coord_list:
            patch_img = self.get_patch_img(img, coord)

            patch_img_list.append(patch_img)

        return patch_img_list


class TiffImg(ImgFunc):

    # class TiffImg():

    def __init__(self, svs_path):

        super().__init__()

        self.tif = tifffile.TiffFile(svs_path)

        self.level_dimensions, self.level_downsamples, self.level = self._get_property()

        self.slide_name = os.path.splitext(os.path.basename(svs_path))[0]

    def _get_property(self):

        '''

        获取常用属性



        :return: 图片维度列表、缩放比例列表、页列表

        '''

        shape = []

        level_ds_list = []

        page_list = []

        for i, page in enumerate(self.tif.pages):

            #             print(str(page.compression)) # 如果读不到图请去掉注释查看编码格式。

            if str(page.compression) == 'COMPRESSION.APERIO_JP2000_RGB' or 'COMPRESSION.JPEG':  # 如果读不到图检查编码格式。更换

                shape.append(page.shape[:2])

                level_ds_list.append(shape[0][0] / page.shape[:2][0])

                page_list.append(int(i))

        return shape, level_ds_list, page_list

    def get_img_ds(self, roi_level_ds):

        """

        读取缩略图



        :param roi_level_ds: 缩小倍数

        :return: 缩略图

        """

        def get_page_ds(roi_level_ds):

            for i, level_ds in enumerate(self.level_downsamples):

                if int(level_ds) <= int(roi_level_ds):
                    continue

                return self.level[i - 1], roi_level_ds / int(self.level_downsamples[i - 1])

            return self.level[-1], roi_level_ds / int(self.level_downsamples[-1])

        page, level_ds = get_page_ds(roi_level_ds)

        img = self.tif.pages[page].asarray()

        if level_ds != 1.:
            resize_img = cv2.resize(img, (0, 0), fx=1 / level_ds, fy=1 / level_ds, interpolation=cv2.INTER_LANCZOS4)

            #             resize_img = cv2.resize(img, (0, 0), fx=1 / level_ds, fy=1 / level_ds, interpolation=cv2.INTER_AREA)

            return resize_img

        return img

    def get_coord_img(self, coord):

        """

        切coord坐标内的图



        :param coord: [[x_min,y_min],[x_max,y_max]]

        :return: 返回图片

        """

        img_ds = self.get_img_ds(1)

        coord_img = self.get_patch_img(img_ds, coord)

        return coord_img

    def get_coord_img_list(self, coord_list):

        '''

        读取一批图片



        :param coord_list: [[[xmin,ymin],[xmax,ymax]],

                            [[xmin,ymin],[xmax,ymax]],...]

        :return: 图片列表

        '''

        img_ds = self.get_img_ds(1)

        img_list = self.get_patch_img_list(img_ds, coord_list)

        return img_list

    def get_cls_img_list(self, coord_list):

        """

        投票分类数据



        :param coord_list:

        :return:

        """

        img_list = []

        for coords in coord_list:
            img_list.append(self.get_coord_img_list(coords))

        return img_list

    def get_img_ds_shape(self, level_ds):

        """

        获取缩小后的图片宽高



        :param level_ds: 缩放比例

        :return: 宽、高

        """

        w0, h0 = self.level_dimensions[0]

        w_ds = int(w0 / level_ds)

        h_ds = int(h0 / level_ds)

        return h_ds, w_ds


class OpenSlideImg():

    def __init__(self, svs_path):

        self.slide = openslide.OpenSlide(svs_path)

        self.slide_name = os.path.splitext(os.path.basename(svs_path))[0]

    def get_img_ds(self, level_ds):

        '''

        读取缩略图



        :param level_ds: 缩小倍数

        :return: 缩略图

        '''

        h_ds, w_ds = self.get_img_ds_shape(level_ds)

        img_ds = np.array(self.slide.get_thumbnail((w_ds, h_ds)))[:, :, :3]

        img_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)

        return img_ds

    def get_coord_img(self, coord, level_ds=1):

        '''

        切coord坐标内的图



        :param coord: [[x_min,y_min],[x_max,y_max]]

        :return: 返回图片

        '''

        def get_page_ds(roi_level_ds):

            print(self.slide.level_downsamples)

            for i, level_ds in enumerate(self.slide.level_downsamples):

                if int(level_ds) <= int(roi_level_ds):
                    continue

                return i - 1, int(level_ds) / roi_level_ds

            return self.slide.level_count - 1, roi_level_ds / int(self.slide.level_downsamples[-1])

        if level_ds == 1:

            img = np.array(self.slide.read_region(coord[0], 0, (coord[1][0] - coord[0][0], coord[1][1] - coord[0][1])))

        else:

            page, resize_ds = get_page_ds(level_ds)

            img = np.array(

                self.slide.read_region(coord[0], page, (coord[1][0] - coord[0][0], coord[1][1] - coord[0][1])))

            img = cv2.resize(img, (0, 0), fx=1 / resize_ds, fy=1 / resize_ds, interpolation=cv2.INTER_LANCZOS4)

        return img

    def get_coord_img_list(self, coord_list):

        '''

        读取一批图片



        :param coord_list: [[[xmin,ymin],[xmax,ymax]],

                            [[xmin,ymin],[xmax,ymax]],...]

        :return: img_list

        '''

        cv_img_list_lv0 = []

        for coord in coord_list:
            img = self.get_coord_img(coord)

            cv_img_list_lv0.append(img)

        return cv_img_list_lv0

    def get_cls_img_list(self, coord_list):

        """

        投票分类数据



        :param coord_list:

        :return:

        """

        img_list = []

        for coord in coord_list:
            img_list.append(self.get_coord_img_list(coord))

        return img_list

    def get_img_ds_shape(self, level_ds):

        """

        获取缩小后的图片宽高



        :param level_ds: 缩放比例

        :return: 高、宽

        """

        w0, h0 = self.slide.level_dimensions[0]

        w_ds = int(w0 / level_ds)

        h_ds = int(h0 / level_ds)

        return h_ds, w_ds

    def get_AppMag_MPP(self):

        MaxMag = eval(self.slide.properties['openslide.objective-power'])

        MPP = round(eval(self.slide.properties['openslide.mpp-x']), 4)

        return MaxMag, MPP
