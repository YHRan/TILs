# -*- encoding: utf-8 -*-
"""
@File    : st_json.py
@Time    : 2021/10/26 17:49
@Author  : 高中需
@Usage   : ReadJson -> 读json、WriteJson -> 写json、AddJson -> 绘制json缩略图（单张）、batch_add -> 批量绘制json缩略图
"""

import os
import cv2
import csv
import json
import glob
import numpy as np

try:
    from tool.read_img import OpenSlideImg, TiffImg
except:
    print('add json need read_img')
    pass


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

    def save_label_data(self,save_dir):
        """
        保存类别信息

        :param save_dir: 保存地址
        """
        label_dict = self.get_label_num()
        rows = list(label_dict.items())
        headers = ['label', 'num']
        save_path = os.path.join(save_dir,self.json_name)+'.csv'
        def save_csv(headers, rows, save_path):
            with open(save_path, 'w', newline='')as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(rows)
        save_csv(headers,rows,save_path)


class WriteJson():
    #save_dir保存路径，slide_path为svs路径
    def __init__(self, save_dir, slide_path, data_dict: dict):
        """
        用于写json

        :param save_dir: 保存文件夹
        :param slide_name: 切片名
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
        #slide_path为svs路径
        self.slide_path = slide_path
        self.slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        # save_dir保存路径
        self.save_path = os.path.join(save_dir, self.slide_name) + '.json'
        self.data_dict = data_dict
        # if svs_path:
        #     try:
        #         self.imgMemorySize = os.stat(svs_path).st_size
        #     except:
        #         self.imgMemorySize = None

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

                self.skeleton['_via_img_metadata'][os.path.basename(self.slide_path)]['regions'].append(one_cnt_json)
                id += 1

    def main(self):
        self.skeleton['_via_img_metadata'][os.path.basename(self.slide_path)] = {'filename': os.path.basename(self.slide_path), 'size': 0,
                                                               'regions': []}
        self.fill_json()
        self.save_json(savepath=self.save_path, json_content=self.skeleton)


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
        self.color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 0), (0, 255, 255)]

    def single_add(self, level_ds, thickness, label_dict: dict):
        """
        单张绘制展示图

        :param level_ds: 缩放比例
        :param thickness: 绘图线宽
        :param label_dict: 制定类别和颜色

        :return:
        """
        img_ds = self.read_svs.get_img_ds(level_ds)
        img_rgb_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)
        json_label = self.read_json.get_json_label()
        if label_dict:
            for label, color in label_dict.items():
                conts_ds1 = self.read_json.get_conts_list(label=label)
                conts = self.read_json.transform_conts_level(conts_ds1, 1, roi_level_ds=level_ds)
                cv2.drawContours(img_rgb_ds, conts, -1, color, thickness)
        else:
            for i, label in enumerate(json_label):
                print(label, self.color_list[i])
                conts_ds1 = self.read_json.get_conts_list(label=label)
                conts = self.read_json.transform_conts_level(conts_ds1, 1, roi_level_ds=level_ds)
                cv2.drawContours(img_rgb_ds, conts, -1, self.color_list[i], thickness)
        cv2.imwrite(self.save_path, img_rgb_ds)


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
        add_json.single_add(level_ds=level_ds, thickness=thickness, label_dict={})


if __name__ == '__main__':
    # json_dir = r'C:\Users\admin\Desktop\out_conts'
    # svs_dir = r'C:\Users\admin\Desktop\out_conts'
    # save_dir = r'C:\Users\admin\Desktop\out_conts'
    # batch_add(json_dir, svs_dir, save_dir, level_ds=4, thickness=3)

    # json_path = r'E:\liver_vac\Vacuole_1124\json\AI202105_R17-xxxx-RD_19-00048-3 2M.json'
    # svs_path = r'E:\liver_vac\svs\AI202105_R17-xxxx-RD_19-00048-3 2M.svs'
    # # save_dir = r'E:\kidney_V2\fix\assess_model\debug'
    # save_dir = r'E:\liver_vac\Vacuole_1124\debug'
    # # read_json = ReadJson(json_path)
    # # print(read_json.get_label_num())
    # # label_dict = {'GM': (255, 255, 0), 'GM_Lesion1': (0, 0, 0), 'GM_Lesion2': (0, 0, 255)}
    # add_json = AddJson(json_path, svs_path, save_dir)
    # add_json.single_add(level_ds=4, thickness=1, label_dict={})


    json_path =r'D:\data\oophoron\debug\debug_1\ST18Rf-OS-063-4-00109.json'
    read_json = ReadJson(json_path)
    center_list_ds1 = read_json.get_conts_list('oop')
    read_img = TiffImg('D:\data\oophoron\svs\ST18Rf-OS-063-4-00109.svs')
    img_ds4 = read_img.get_img_ds(4)
    for center_ds1 in center_list_ds1:
        center_ds4 = np.array(center_ds1.reshape(-1,2)/4,dtype='int32')[0]
        print(center_ds4)
        cv2.circle(img_ds4, tuple(center_ds4), radius=4, color=(0, 0, 255), thickness=-1)

    cv2.imwrite('img.png',img_ds4)