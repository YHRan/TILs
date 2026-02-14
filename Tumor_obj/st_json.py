# -*- encoding: utf-8 -*-
"""
@File    : st_json.py
@Time    : 2021/10/26 17:49
@Author  : 高中需
@Usage   : ReadJson -> 读json; WriteJson -> 写json; AddJson -> 绘制json缩略图（单张）; batch_add -> 批量绘制json缩略图; MergeJson -> 合并json
"""

import os
import cv2
import csv
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import pdb
import openslide
import tifffile


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
        return img_ds

    def get_img_ds_with_zangqi(self, level_ds, json_path):
        """
        读取缩略图

        :param level_ds: 缩小倍数
        :return: 缩略图
        """
        h_ds, w_ds = self.get_img_ds_shape(level_ds)
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            (ds1_x1, ds1_y1), (ds1_x2, ds1_y2) = json_data[self.slide_name]
        x1 = ds1_x1 // level_ds
        y1 = ds1_y1 // level_ds
        x2 = ds1_x2 // level_ds
        y2 = ds1_y2 // level_ds

        img_ds = np.array(self.slide.get_thumbnail((w_ds, h_ds)))[y1:y2, x1:x2, :3]  # 切片似乎没必要
        return img_ds, (ds1_x1, ds1_y1)

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
        w_ds = int(w0 / level_ds + 0.5)
        h_ds = int(h0 / level_ds + 0.5)
        return h_ds, w_ds

    def get_AppMag_MPP(self):
        MaxMag = eval(self.slide.properties['openslide.objective-power'])
        MPP = round(eval(self.slide.properties['openslide.mpp-x']), 4)
        return MaxMag, MPP


class ReadJson():
    def __init__(self, json_path):
        """
        读取json

        :param json_path: json地址
        """
        self.d_json = json.loads(open(json_path, "r", encoding="utf8").read())['_via_img_metadata']
        self.json_name = os.path.splitext(os.path.basename(json_path))[0]

    def yield_json_data(self, label):
        for key in self.d_json:
            data = self.d_json.get(key)
            for region in data['regions']:
                if region['region_attributes']['bone_marrow'] == label:
                    x_y_list = np.array(np.vstack(
                        (region["shape_attributes"]["all_points_x"],
                         region["shape_attributes"]["all_points_y"])).T, dtype='int32').reshape((-1, 1, 2))
                    yield x_y_list

    def get_conts_list(self, label):
        """
        获取指定类别的轮廓
        """
        conts_ds1 = []
        for x_y_list in self.yield_json_data(label):
            conts_ds1.append(x_y_list)
        return conts_ds1

    def get_boxs(self, label):
        """
        获取指定类别的矩形框
        """
        boxs_ds1 = []
        for x_y_list in self.yield_json_data(label):
            box_min = np.min(x_y_list, axis=0)
            box_max = np.max(x_y_list, axis=0)
            boxs_ds1.append(np.array([box_min[0], box_max[0]], dtype='int32'))
        return np.array(boxs_ds1, dtype='int32')

    def change_ds(self, data, data_ds, roi_ds):
        return np.array(data * data_ds / roi_ds, dtype='int32')

    def get_json_label(self):
        """
        获取类别标签
        """
        json_list = []
        for key in self.d_json:
            data = self.d_json.get(key)
            for region in data['regions']:
                json_list.append(region['region_attributes']['bone_marrow'])
        json_label_list = sorted(list(set(json_list)))
        return json_label_list

    def get_label_num(self):
        """
        获取各类别数量
        """
        label_num_dict = {}
        for label in self.get_json_label():
            label_num_dict[label] = 0
        for key in self.d_json:
            data = self.d_json.get(key)
            for region in data['regions']:
                label_num_dict[region['region_attributes']['bone_marrow']] += 1
        return label_num_dict

    def transform_conts_level(self, conts, conts_level_ds, roi_level_ds=1):
        """
        轮廓修改缩放比例

        :param conts: 轮廓
        :param conts_level_ds: 轮廓的缩放比例
        :param roi_level_ds: 目标缩放比例
        """
        level_ds = conts_level_ds
        new_conts = []
        for cont in conts:
            new_conts.append((cont * level_ds / roi_level_ds).astype('int'))
        return new_conts

    def save_label_data(self, save_dir):
        """
        保存类别信息

        :param save_dir: 保存地址
        """
        label_dict = self.get_label_num()
        rows = list(label_dict.items())
        headers = ['label', 'num']
        save_path = os.path.join(save_dir, self.json_name) + '.csv'

        def save_csv(headers, rows, save_path):
            with open(save_path, 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(rows)

        save_csv(headers, rows, save_path)


class WriteJson():
    def __init__(self, save_path, svs_path, data_dict: dict):
        """
        用于写json

        :param save_dir: 保存路径。。。
        :param svs_path: 图片或者svs地址
        :param data_dict: {'cv':cv_conts,'pa':pa_conts}
        """
        self.skeleton = {
            '_via_settings': {
                'ui': {
                    'annotation_editor_height': 25,
                    'annotation_editor_fontsize': 0.8,
                    'leftsidebar_width': 18,
                    'image_grid': {
                        'img_height': 80,
                        'rshape_fill': 'none',
                        'rshape_fill_opacity': 0.3,
                        'rshape_stroke': 'yellow',
                        'rshape_stroke_width': 2,
                        'show_region_shape': True,
                        'show_image_policy': 'all'},
                    'image': {
                        'region_label': 'region_id',
                        'region_label_font': '10px Sans'}},
                'core': {
                    'buffer_size': 18,
                    'file_path': {},
                    'default_filepath': ''},
                'project': {
                    'name': ''}},
            '_via_img_metadata': {},
            '_via_attributes': {
                'region': {'bone_marrow': {'type': 'text'}},
                'file': {}}
        }
        self.slide_name = os.path.splitext(os.path.basename(svs_path))[0]
        self.save_path = save_path
        self.data_dict = data_dict
        if svs_path:
            try:
                self.imgMemorySize = os.stat(svs_path).st_size
            except:
                self.imgMemorySize = None

    def save_json(self, json_content, savepath):
        with open(savepath, 'w') as file:
            json.dump(json_content, file)

    def fill_json(self):
        id = 0
        for label, conts in self.data_dict.items():
            for cnt in conts:
                all_points_x = []
                all_points_y = []
                for coord in cnt:
                    x = float(coord[0][0])
                    y = float(coord[0][1])
                    all_points_x.append(x)
                    all_points_y.append(y)

                one_cnt_json = {'shape_attributes': {'name': 'polygon', 'anno_id': id,
                                                     'all_points_x': all_points_x, 'all_points_y': all_points_y},
                                'region_attributes': {'bone_marrow': label}}

                self.skeleton['_via_img_metadata'][self.slide_name]['regions'].append(one_cnt_json)
                id += 1

    def main(self):
        self.skeleton['_via_img_metadata'][self.slide_name] = {'filename': self.slide_name + '.svs',
                                                               'size': self.imgMemorySize,
                                                               'regions': []}
        self.fill_json()
        self.save_json(savepath=self.save_path, json_content=self.skeleton)


class WriteJsonV2():
    def __init__(self, save_dir, json_name, data_dict: dict):
        """
        用于写json

        :param save_dir: 保存文件夹
        :param json_name: 保存json名字，不带后缀
        :param data_dict: {'cv':cv_conts,'pa':pa_conts}
        """
        self.skeleton = {
            '_via_settings': {
                'ui': {
                    'annotation_editor_height': 25,
                    'annotation_editor_fontsize': 0.8,
                    'leftsidebar_width': 18,
                    'image_grid': {
                        'img_height': 80,
                        'rshape_fill': 'none',
                        'rshape_fill_opacity': 0.3,
                        'rshape_stroke': 'yellow',
                        'rshape_stroke_width': 2,
                        'show_region_shape': True,
                        'show_image_policy': 'all'},
                    'image': {
                        'region_label': 'region_id',
                        'region_label_font': '10px Sans'}},
                'core': {
                    'buffer_size': 18,
                    'file_path': {},
                    'default_filepath': ''},
                'project': {
                    'name': ''}},
            '_via_img_metadata': {},
            '_via_attributes': {
                'region': {'bone_marrow': {'type': 'text'}},
                'file': {}}
        }
        self.json_name = json_name
        self.save_path = os.path.join(save_dir, self.json_name) + '.json'
        self.data_dict = data_dict

    def save_json(self, json_content, savepath):
        with open(savepath, 'w') as file:
            json.dump(json_content, file)

    def fill_json(self):
        id = 0
        for label, conts in self.data_dict.items():
            for cnt in conts:
                all_points_x = []
                all_points_y = []
                for coord in cnt:
                    x = float(coord[0][0])
                    y = float(coord[0][1])
                    all_points_x.append(x)
                    all_points_y.append(y)

                one_cnt_json = {'shape_attributes': {'name': 'polygon', 'anno_id': id,
                                                     'all_points_x': all_points_x, 'all_points_y': all_points_y},
                                'region_attributes': {'bone_marrow': label}}

                self.skeleton['_via_img_metadata'][self.json_name]['regions'].append(one_cnt_json)
                id += 1

    def main(self):
        self.skeleton['_via_img_metadata'][self.json_name] = {'filename': '', 'size': 0,
                                                              'regions': []}
        self.fill_json()
        self.save_json(savepath=self.save_path, json_content=self.skeleton)


class MergeJson():
    def __init__(self, json_list):
        self.json_list = json_list

    def merge_json(self):
        pass

    def main(self, labels, save_dir):
        json_name = os.path.splitext(os.path.basename(self.json_list[0]))[0]
        if labels == None:
            label_set = set()
            for json_path in self.json_list:
                for label in ReadJson(json_path).get_json_label():
                    label_set.add(label)
            print(label_set)  # Todo: 没实现呢
        else:
            data_dict = {}
            for label in labels:
                conts = []
                for json_path in self.json_list:
                    conts += ReadJson(json_path).get_conts_list(label)
                data_dict[label] = conts

            WriteJsonV2(save_dir, json_name, data_dict).main()


class AddJson():
    def __init__(self, json_path, svs_path, save_dir):
        """
        绘制缩略图

        :param json_path: json地址
        :param svs_path: svs地址
        :param save_dir: 保存地址
        """
        self.read_json = ReadJson(json_path)
        self.read_svs = OpenSlideImg(svs_path)
        #         self.read_svs = TiffImg(svs_path)
        # self.img_path = img_path
        self.save_path = os.path.join(save_dir, self.read_svs.slide_name) + '.png'
        # self.save_path = os.path.join(save_dir+self.img_path.split('\\')[-1])
        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 0), (0, 255, 255),
                           (0, 175, 255)]

    def translucence(self, img_rgb_ds, conts, color):
        mask_2 = img_rgb_ds.copy()
        cv2.drawContours(mask_2, conts, -1, color, -1)
        result_img = cv2.addWeighted(img_rgb_ds, 0.78, mask_2, 0.22, 1)
        return result_img

    def single_add(self, level_ds, thickness, label_color_dict: dict = None, translucence_label_list: list = None):
        """
        单张绘制展示图

        :param level_ds: 缩放比例
        :param thickness: 绘图线宽
        :param label_color_dict: 制定类别和颜色

        :return:
        """
        img_ds = self.read_svs.get_img_ds(level_ds)
        #         print(img_ds.shape())
        #         img_rgb_ds = img_ds
        # img_rgb_ds = cv2.imread(self.img_path)
        img_rgb_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)
        #         cv2.imwrite('img.png', img_rgb_ds)
        json_label = self.read_json.get_json_label()
        if label_color_dict:
            for label, color in label_color_dict.items():
                conts_ds1 = self.read_json.get_conts_list(label=label)
                conts = self.read_json.transform_conts_level(conts_ds1, 1, roi_level_ds=level_ds)
                if label in translucence_label_list:
                    img_rgb_ds = self.translucence(img_rgb_ds, conts, color)
                else:
                    #                     pdb.set_trace()
                    cv2.drawContours(img_rgb_ds, conts, -1, color, thickness)
        else:
            for i, label in enumerate(json_label):
                print(label, self.color_list[i])
                conts_ds1 = self.read_json.get_conts_list(label=label)
                conts = self.read_json.transform_conts_level(conts_ds1, 1, roi_level_ds=level_ds)
                if label in translucence_label_list:
                    img_rgb_ds = self.translucence(img_rgb_ds, conts, self.color_list[i])
                else:
                    cv2.drawContours(img_rgb_ds, conts, -1, self.color_list[i], thickness)
        #         cv2.imwrite(self.save_path, img_rgb_ds)
        return img_rgb_ds


class Analysis():
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_read = ReadJson(json_path)

    def label_area_list(self, label):
        label_conts = self.json_read.get_conts_list(label)
        label_area_list = [cv2.contourArea(cont) for cont in label_conts]
        return label_area_list

    @classmethod
    def plt_hist(cls, label, label_data, bins):
        plt.clf()
        data_list = np.array(label_data)
        plt.hist(data_list, bins, alpha=1)
        plt.xlabel('area')
        plt.ylabel('num')
        plt.title(label)
        # plt.show()
        flg = plt.gcf()
        flg.savefig('debug/' + label + '.png')

def array_to_STAI(img, end_path, page_amount=4, sampel=2):
    with tifffile.TiffWriter(end_path, bigtiff=True, append=True) as tif_new:
        tif_new.save(img,
                     photometric='rgb',
                     compression='jpeg',
                     planarconfig='CONTIG',
                     tile=(256, 256),
                     subsampling=(1, 1),
                     subfiletype=9,
                     datetime=True,
                     metadata={'scanner_model': 'Aperio Leica Biosystems GT450 v1.0.1',
                               'openslide.mpp-x': '0.26283899999999999', 'openslide.mpp-y': '0.26283899999999999'}
                     )
        for i in range(page_amount):
            img = cv2.resize(img, (0, 0), fx=1 / sampel, fy=1 / sampel)
            tif_new.save(img,
                         photometric='rgb',
                         compression='jpeg',
                         planarconfig='CONTIG',
                         tile=(256, 256),
                         subsampling=(1, 1),
                         subfiletype=9,
                         )

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


def match_svs_json(svs_list, json_list):
    """
    匹配 svs和json

    :param svs_list: svs地址列表
    :param json_list: json地址列表
    :return:
    """

    def get_name(path):
        # 获取名字
        return os.path.splitext(os.path.basename(path))[0]

    def match(svs_name, json_name):
        # 匹配规则
        if svs_name[:] == json_name[:]:
            return True
        else:
            return False

    new_svs_list = []
    new_json_list = []
    svs_name_list = [get_name(svs_path) for svs_path in svs_list]
    json_name_list = [get_name(json_path) for json_path in json_list]

    for i, svs_name in enumerate(svs_name_list):
        for j, json_name in enumerate(json_name_list):
            if match(svs_name, json_name):
                new_json_list.append(json_list[j])
                new_svs_list.append(svs_list[i])
    return new_svs_list, new_json_list


def batch_add(json_dir, svs_dir, save_dir, level_ds, thickness):
    """
    批量绘制json结果图
    ！！！无结果请修改match_svs_json内match的匹配规则！！！

    :param json_dir: json保存地址
    :param svs_dir: svs保存地址
    :param save_dir: 绘制图片保存地址
    :param level_ds: 绘图缩放比例
    :param thickness: 绘图线宽
    :return:
    """

    json_list = get_all_file(json_dir, '.json')
    svs_list = get_all_file(svs_dir, '.svs')
    svs_list, json_list = match_svs_json(svs_list, json_list)
    for svs_path, json_path in zip(svs_list, json_list):
        add_json = AddJson(json_path, svs_path, save_dir)
        add_json.single_add(level_ds=level_ds, thickness=thickness, label_color_dict=None, translucence_label_list=[])


if __name__ == '__main__':
    #     label_color = {'large WP': (0, 255, 0), 'small WP': (0, 0, 255),'SP': (255, 0, 0),'central artery': (0, 55, 75)}
    label_color = {'cross_cell': (0, 255, 0), 'vertical_cell': (0, 255, 0), 'cross_blood': (255, 0, 0),
                   'vertical_blood': (255, 0, 0)}
    json_dir = r'/home/jfq/data-jfq/qly/pipline_v1_new/ST15Rm-TE-293-4-00099/'
    json_list = get_all_file(json_dir, '.json')
    svs_dirs = r'/home/data/wsi/ST15R-TE-293_Testis_saomiaowenjian/'
    save_dir = r'/home/jfq/data-jfq/qly/pipline_v1_new/ST15Rm-TE-293-4-00099'
    # label_color_dict = {'PF': (0, 0, 255), 'PFs': (0, 128, 0), 'SF': (255, 165, 0), 'TF': (128, 0, 128),
    #                     'AF': (255, 0, 0), 'VF': (255, 255, 0), 'AF1': (255, 0, 0)}
    for json_path in json_list:
        #         try:
        svs_name = json_path.split('/')[-1].split('.')[0]
        print(svs_name)
        svs_dir = svs_dirs + svs_name + '.svs'
        print(svs_dir)
        add_json = AddJson(json_path, svs_dir, save_dir, )
        img = add_json.single_add(level_ds=1, thickness=2, label_color_dict=label_color,
                                  translucence_label_list=['PA_large_vac'])
        print('add_json over')
        end_path = save_dir + '/' + svs_name + '_finall_liutest.svs'  # change by tangmy
        array_to_STAI(img, end_path)
