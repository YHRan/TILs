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
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

try:
    from read_img import OpenSlideImg, TiffImg
except Exception as e:
    print(e)
    print('add json need read_img')
    pass


class ReadJson():
    def __init__(self, json_path):
        """
        读取json

        :param json_path: json地址
        """
        self.d_json = json.loads(open(json_path, "r", encoding="utf8").read())['_via_img_metadata']
        # KeyError: '_via_img_metadata'
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
            with open(save_path, 'w', newline='')as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(rows)

        save_csv(headers, rows, save_path)


class WriteJson():
    def __init__(self, save_dir, svs_path, data_dict: dict):
        """
        用于写json

        :param save_dir: 保存文件夹
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
        self.save_path = os.path.join(save_dir, self.slide_name) + '.json'
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
        self.skeleton['_via_img_metadata'][self.slide_name] = {'filename': '', 'size': self.imgMemorySize,
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
        self.save_path = os.path.join(save_dir, self.read_svs.slide_name) + '.png'
        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 0), (0, 255, 255),
                           (0, 175, 255)]

    def translucence(self, img_rgb_ds, conts, color):
        mask_2 = img_rgb_ds.copy()
        for cont in conts:
            cv2.drawContours(mask_2, [cont], -1, color, -1)
        result_img = cv2.addWeighted(img_rgb_ds, 0.78, mask_2, 0.22, 1)
        return result_img

    def single_add(self, level_ds, thickness, label_color_dict: dict = None, translucence_label_list: list = []):
        """
        单张绘制展示图

        :param level_ds: 缩放比例
        :param thickness: 绘图线宽
        :param label_color_dict: 制定类别和颜色

        :return:
        """
        img_ds = self.read_svs.get_img_ds(level_ds)
        json_label = self.read_json.get_json_label()
        if label_color_dict:
            for label, color in label_color_dict.items():
                conts_ds1 = self.read_json.get_conts_list(label=label)
                conts = self.read_json.transform_conts_level(conts_ds1, 1, roi_level_ds=level_ds)
                if label in translucence_label_list:
                    img_ds = self.translucence(img_ds, conts, color)
                else:
                    cv2.drawContours(img_ds, conts, -1, color, thickness)
        else:
            for i, label in enumerate(json_label):
                print(label, self.color_list[i])
                conts_ds1 = self.read_json.get_conts_list(label=label)
                conts = self.read_json.transform_conts_level(conts_ds1, 1, roi_level_ds=level_ds)
                if label in translucence_label_list:
                    img_ds = self.translucence(img_ds, conts, self.color_list[i])
                else:
                    cv2.drawContours(img_ds, conts, -1, self.color_list[i], thickness)

        img_rgb_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.save_path, img_rgb_ds)

    def centroid(self, img, conts, name=None):
        """
        绘制质心。
        坐标原点为左下。
        :param img: 图像
        :param conts: 轮廓
        :param name: 保存质心txt文件名（不用后缀）
        :return: 绘制质心后图像
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        str_1 = ''
        img_h, img_w = img.shape[:2]
        for cont in conts:
            x, y, w, h = cv2.boundingRect(cont)
            if cv2.contourArea(cont) < 2:
                continue
            M = cv2.moments(cont)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            cv2.circle(img, tuple([int(center_x), int(center_y)]), 2, (255, 0, 0), 4)
            cv2.putText(img, str((center_x, img_h - center_y)), tuple([x, y]), font, 1, (255, 0, 0), 1)
            str_1 += str((center_x, img_h - center_y)) + '\n'
        if name:
            with open(os.path.join(save_dir, name) + '.txt', "w") as f:
                f.write(str_1)
        return img

    def scale(self, img, num):
        """
        绘制0级坐标轴，原点在左下。
        :param img: 需要绘制的图像。
        :param num: 区间个数
        :return: 图像
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_h, img_w = img.shape[:2]
        cv2.line(img, (50, 50), (50, img_h - 50), (0, 0, 0), 10)  # y轴
        cv2.line(img, (50, 50), (35, 150), (0, 0, 0), 10)
        cv2.line(img, (50, 50), (65, 150), (0, 0, 0), 10)

        cv2.line(img, (50, img_h - 50), (img_w - 50, img_h - 50), (0, 0, 0), 10)  # x轴
        cv2.line(img, (img_w - 50, img_h - 50), (img_w - 150, img_h - 65), (0, 0, 0), 10)
        cv2.line(img, (img_w - 50, img_h - 50), (img_w - 150, img_h - 35), (0, 0, 0), 10)

        cv2.putText(img, str((0, 0)), (80, img_h - 100), font, 2, (0, 0, 0), 5)

        for i in range(1, num):
            x = i * int(img_w / num)
            y = i * int(img_h / num)
            cv2.line(img, (50, img_h - y), (100, img_h - y), (0, 0, 0), 8)
            cv2.putText(img, str(y), (110, img_h - y), font, 2, (0, 0, 0), 5)

            cv2.line(img, (x, img_h - 50), (x, img_h - 100), (0, 0, 0), 8)
            cv2.putText(img, str(x), (x, img_h - 110), font, 2, (0, 0, 0), 5)

        return img


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
    assert len(new_svs_list)==len(new_json_list)
    print('matched:',len(new_json_list))
    return new_svs_list, new_json_list


def batch_add(json_dir, svs_dir, save_dir, level_ds, label_color_dict, thickness, translucence_label_list):
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
        add_json.single_add(level_ds=level_ds, thickness=thickness, label_color_dict=label_color_dict,
                            translucence_label_list=translucence_label_list)


def read_txt(txt_path):
    list_ = []
    with open(txt_path, 'r') as f:
        while True:
            a = f.readline()
            list_.append(a[:-1])
            if a == '':
                break


if __name__ == '__main__':
    label_color_dict = {'G': (0, 255, 0)}

    all_json = glob.glob(r'D:\ProjectSpace\STRUCTURE_DEV\LG\prepare_dataset_cell\dataset\json\*.json')
    all_svs = glob.glob(r'D:\ProjectSpace\STRUCTURE_DEV\LG\prepare_dataset_cell\dataset\svs\*.svs')
    save_dir = r'D:\ProjectSpace\STRUCTURE_DEV\LG\prepare_dataset_cell\vis_json'
    print(len(all_json),len(all_svs))
    for json_dir,svs_dir in zip(all_json,all_svs):
        name=json_dir.split('\\')[-1].split('.')[0]+'.png'
        read = ReadJson(json_dir)
        conts = read.get_conts_list('cell')
        read_svs = OpenSlideImg(svs_dir)
        img=read_svs.get_img_ds(1)
        h_ds, w_ds = read_svs.get_img_ds_shape(1)
        mask = np.zeros((h_ds,w_ds)).astype(np.uint8)
        cv2.drawContours(mask,conts,-1,(255),2)
        # mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        # cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # lable_list = {'cell':cnts}
        # WriteJson(save_dir,svs_dir,lable_list).main()
        # mask = np.zeros((h_ds,w_ds)).astype(np.uint8)
        cv2.drawContours(img,conts,-1,(255),2)
        cv2.imwrite(os.path.join(save_dir,name),img)