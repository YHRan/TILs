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
# import tifffile
import openslide
import numpy as np
from multiprocessing import Pool


import tifffile


def array_to_STAI(img, end_path, page_amount=4, sampel=2):
    with tifffile.TiffWriter(end_path, bigtiff=True, append=True) as tif_new:
        tif_new.save(img,
                     photometric='rgb',
                     compress='jpeg',
                     planarconfig='CONTIG',
                     tile=(256, 256),
                     #                      subsampling=(1, 1),
                     subfiletype=9,
                     datetime=True,
                     metadata={'scanner_model': 'Aperio Leica Biosystems GT450 v1.0.1',
                               'openslide.mpp-x': '0.26283899999999999',
                               'openslide.mpp-y': '0.26283899999999999',
                               'openslide.objective-power': '40'}

                     )
        for i in range(page_amount):
            img = cv2.resize(img, (0, 0), fx=1 / sampel, fy=1 / sampel)
            tif_new.save(img,
                         photometric='rgb',
                         compress='jpeg',
                         planarconfig='CONTIG',
                         tile=(256, 256),
                         #                          subsampling=(1, 1),
                         subfiletype=9,
                         )

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
            # print(str(page.compression)) # 如果读不到图请去掉注释查看编码格式。
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

    def get_page_ds(self, roi_level_ds):
        for i, level_ds in enumerate(self.slide.level_downsamples):
            if int(level_ds) <= int(roi_level_ds):
                continue
            return i - 1, self.slide.level_downsamples[i - 1], roi_level_ds / int(self.slide.level_downsamples[i - 1])
        return len(self.slide.level_downsamples) - 1, self.slide.level_downsamples[-1], roi_level_ds / int(
            self.slide.level_downsamples[-1])

    def get_img_ds(self, level_ds):
        '''
        读取缩略图

        :param level_ds: 缩小倍数
        :return: 缩略图
        '''
        h_ds, w_ds = self.get_img_ds_shape(level_ds)
        img_ds = np.array(self.slide.get_thumbnail((w_ds, h_ds)))[:, :, :3]
        return img_ds

    def get_coord_img(self, coord):
        '''
        切coord坐标内的图

        :param coord: [[x_min,y_min],[x_max,y_max]]
        :return: 返回图片
        '''
        img = np.array(self.slide.read_region(coord[0], 0, (coord[1][0] - coord[0][0], coord[1][1] - coord[0][1])),
                       dtype='uint8')
        return img

    def get_coord_img_ds(self, coord, level_ds):
        '''
        切coord坐标内的图

        :param coord: [[x_min,y_min],[x_max,y_max]]
        :level_ds: 下采样等级
        :return: 返回图片
        '''
        page, page_ds, resize_ds = self.get_page_ds(level_ds)
        point = np.array(coord[0], dtype=np.int32) * level_ds
        size = np.array(coord[1], dtype=np.int32) - np.array(coord[0], dtype=np.int32)
        size = [int(size[0] * resize_ds), int(size[1] * resize_ds)]
        img = np.array(self.slide.read_region(point, page, tuple(size)), dtype='uint8')
        img_resize = cv2.resize(img, (0, 0), fx=1 / resize_ds, fy=1 / resize_ds)
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        return img_resize

    def get_coord_img_list(self, coord_list):
        '''
        读取一批图片

        :param coord_list: [[[xmin,ymin],[xmax,ymax]],
                            [[xmin,ymin],[xmax,ymax]],...]
        :return: 图片列表
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

def mp_get_coord_img_list(svs_path, coord_list):
    """
    多进程读图 内部使用openslide接口

    :param svs_path: svs地址
    :param coord_list: [[[xmin,ymin],[xmax,ymax]],
                        [[xmin,ymin],[xmax,ymax]],...]
    :return: img_list
    """

    def group(lists, procees):
        split_list = []
        length = len(lists)
        for i in range(procees):
            one_list = lists[math.floor(i / procees * length):math.floor((i + 1) / procees * length)]
            split_list.append(one_list)
        return split_list

    def read_image_(coord_list, svs_path):
        read_svs = OpenSlideImg(svs_path)
        img_list_ds1 = read_svs.get_coord_img_list(coord_list)
        return img_list_ds1

    pool_num = 16
    data_grou_list = group(coord_list, procees=pool_num)
    pool = Pool(pool_num)
    ret_list = []
    for i in range(pool_num):
        ret = pool.apply_async(read_image_, (data_grou_list[i], svs_path), )
        ret_list.append(ret)
    pool.close()
    pool.join()
    All_image_data = []
    for ret in ret_list:
        data_array_list = ret.get()
        for imag in data_array_list:
            All_image_data.append(imag)
    return All_image_data


if __name__ == '__main__':
    # svs_path = r'C:\Users\admin\Desktop\AITC202103_R19-xxxx-RD_20000001-3 1M.svs'
    # read_svs = OpenSlideImg(svs_path)
    # img_ds16 = read_svs.get_img_ds(16)
    # print(img_ds16.shape)
    # print(read_svs.get_img_ds_shape(16))

    svs_path = r'D:\Progrems\nephritic\wsi_sn_img_0404L_2.svs'
    read_svs = TiffImg(svs_path)
    print(read_svs.level_dimensions)
    print(read_svs.level_downsamples)
