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



# def array_to_STAI(img, end_path, page_amount=4, sampel=2):
#     with tifffile.TiffWriter(end_path, bigtiff=True, append=True) as tif_new:
#         tif_new.save(img,
#                      photometric='rgb',
#                      compress='jpeg',
#                      planarconfig='CONTIG',
#                      tile=(256, 256),
#                      subsampling=(1, 1),
#                      subfiletype=9,
#                      )
#         for i in range(page_amount):
#             img = cv2.resize(img, (0, 0), fx=1 / sampel, fy=1 / sampel)
#             tif_new.save(img,
#                          photometric='rgb',
#                          compress='jpeg',
#                          planarconfig='CONTIG',
#                          tile=(256, 256),
#                          subsampling=(1, 1),
#                          subfiletype=9,
#                          )

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
        patch_w, patch_h = np.array(coord[1]) - np.array(coord[0])
        patch_img = img[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]

        if patch_img.shape[:2] != (patch_h, patch_w):
            img_h, img_w = patch_img.shape[:2]
            if len(img.shape)==3:
                mask = np.ones((patch_h, patch_w, 3), dtype='uint8') * 255
            else:
                mask = np.ones((patch_h, patch_w), dtype='uint8') * 255

            mask[0:img_h, 0:img_w] = patch_img
            patch_img = mask
        return patch_img


    def get_patch_img_list(self, img, coord_list):
        """
        读取一批图片

        :param img: 原始图
        :param coord_list: [[[xmin,ymin],[xmax,ymax]],
                            [[xmin,ymin],[xmax,ymax]],...]
        :return:
        """

        # #方案一未过滤
        # # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        patch_img_list = []
        for i,coord in enumerate(coord_list):
            # print('coord',coord)
            # exit()
            patch_img = self.get_patch_img(img, coord)
            
            patch_img_list.append(patch_img)
#             cv2.imwrite(f'/home/lyb/data_yhr/03-yhr/External_project/yolo7_pre/test_png/{i}.png', patch_img)

        # dict_box_list_new_now = []
        # dict_box_list_new_now_1 = []
        # for j in luteum_conts_ds:
        #     for i in range(len(coord_list)):
        #         dst0_sum = []
        #         x1 = coord_list[i][0][0]
        #         y1 = coord_list[i][0][1]
        #         x2 = coord_list[i][1][0]
        #         y2 = coord_list[i][0][1]
        #         x3 = coord_list[i][1][0]
        #         y3 = coord_list[i][1][1]
        #         x4 = coord_list[i][0][0]
        #         y4 = coord_list[i][1][1]
        #
        #         dst0 = cv2.pointPolygonTest(j, (x1, y1), False)
        #         dst1 = cv2.pointPolygonTest(j, (x2, y2), False)
        #         dst2 = cv2.pointPolygonTest(j, (x3, y3), False)
        #         dst3 = cv2.pointPolygonTest(j, (x4, y4), False)
        #
        #         # print(dst0,dst1,dst2,dst3)
        #
        #         dst0_sum.append(dst0)
        #         dst0_sum.append(dst1)
        #         dst0_sum.append(dst2)
        #         dst0_sum.append(dst3)
        #
        #         is_a = 0
        #         for k in dst0_sum:
        #             if k < 0:
        #                 is_a = -1
        #                 break
        #             if k >= 0:
        #                 is_a = 1
        #
        #         if is_a == 1:
        #             dict_box_list_new_now.append(i)
        #
        #
        # for li in dict_box_list_new_now:
        #     if li not in dict_box_list_new_now_1:
        #         dict_box_list_new_now_1.append(li)
        #
        # new_box_all_id = []
        # for i in range(len(coord_list)):
        #     new_box_all_id.append(i)
        #
        # for j in dict_box_list_new_now_1:
        #     for k in new_box_all_id:
        #         if j == k:
        #             new_box_all_id.remove(k)
        #
        # new_box_all_xin = []
        # for i in new_box_all_id:
        #     new_box_all_xin.append(coord_list[i])
        #
        # patch_img_list = []
        # for i,coord in enumerate(new_box_all_xin):
        #     patch_img = self.get_patch_img(img, coord)
        #     # cv2.imwrite(f'C:/Users/Administrator/Desktop/2/bug_img{i}.png', patch_img)
        #     patch_img_list.append(patch_img)
        # # # exit()


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
        :return: 高、宽
        """

        h0, w0 = self.level_dimensions[0]
        w_ds = int(w0 / level_ds)
        h_ds = int(h0 / level_ds)
        return h_ds, w_ds

    def get_AppMag_MPP(self):
        AppMag = None
        MPP = None
        a = self.tif.pages._keyframe.description
        for i in a.split('|'):
            if 'AppMag' in i:
                AppMag = eval(i.split('=')[-1])
            if 'MPP' in i:
                MPP = eval(i.split('=')[-1])
        return AppMag, MPP


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
        img_ds_1 = cv2.cvtColor(img_ds, cv2.COLOR_RGB2BGR)

        return img_ds_1

    def get_coord_img(self, coord):
        '''
        切coord坐标内的图

        :param coord: [[x_min,y_min],[x_max,y_max]]
        :return: 返回图片
        '''
        img = np.array(self.slide.read_region(coord[0], 0, (coord[1][0] - coord[0][0], coord[1][1] - coord[0][1])),dtype='uint8')
        return img

    # def get_patch_img_list(self, img, coord_list):
    #     """
    #     读取一批图片
    #
    #     :param img: 原始图
    #     :param coord_list: [[[xmin,ymin],[xmax,ymax]],
    #                         [[xmin,ymin],[xmax,ymax]],...]
    #     :return:
    #     """
    #
    #     # #方案一未过滤
    #     # # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #     patch_img_list = []
    #     for i,coord in enumerate(coord_list):
    #         # print('coord',coord)
    #         # exit()
    #         patch_img = self.get_patch_img(img, coord)
    #         # cv2.imwrite(f'C:/Users/Administrator/Desktop/33/bug_img{i}.png', patch_img)
    #         patch_img_list.append(patch_img)

    def get_coord_img_list(self, coord_list):
        '''
        读取一批图片

        :param coord_list: [[[xmin,ymin],[xmax,ymax]],
                            [[xmin,ymin],[xmax,ymax]],...]
        :return: 图片列表
        '''
        cv_img_list_lv0 = []
        for coord in coord_list:
            # print('coord',coord)
            # exit()
            img = self.get_coord_img(coord)
            img = cv2.cvtColor(img,cv2.COLOR_RGBA2RGB)
            # print(img.shape)
            cv2.imwrite(r'C:\Users\Administrator\Desktop\yc_img/bug_img.png',img)
            # # exit()
            cv_img_list_lv0.append(img)
        return cv_img_list_lv0

    def get_cls_img_list(self, coord_list):
        """
        投票分类数据

        :param coord_list:
        :return:
        """
        cv2.imwrite()
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


def mp_get_coord_img_list(svs_path, coord_list):
    """
    多进程读图 内部使用openslide接口

    :param svs_path: svs地址
    :param coord_list: [[[xmin,ymin],[xmax,ymax]],
                        [[xmin,ymin],[xmax,ymax]],...]
    :return: img_list
    """

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
    svs_path = r'D:\ST18Rf-OS-063-1-00029_new.svs'
    tiff_read = TiffImg(svs_path)
    openslide_read = OpenSlideImg(svs_path)
    print(tiff_read.get_img_ds_shape(1))
    print(openslide_read.get_img_ds_shape(1))
