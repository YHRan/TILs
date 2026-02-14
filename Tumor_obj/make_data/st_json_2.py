import os
import cv2
import json
import datetime
import numpy as np
import random
import csv

try:
    from read_img import OpenSlideImg, TiffImg, array_to_STAI
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
        self.d_json = json.loads(open(json_path, "r", encoding="utf8").read())
        self.json_name = os.path.splitext(os.path.basename(json_path))[0]

    def yield_json_data(self, label):

        for region in self.d_json['features']:

            if region['properties']['label_name'] == label:

                type = list(region['geometry'].values())[0]
                ori_data = list(region['geometry'].values())[1]
                if type == 'Point':
                    data = (np.array(ori_data) * [1, -1]).tolist()
                elif type == 'Line':
                    data = (np.array(ori_data) * [1, -1]).tolist()
                elif type == 'Polygon':
                    if isinstance(ori_data, list):  # 是list的话判断为内嵌结构
                        data = []
                        for temp_data in ori_data:
                            temp_data = np.array(temp_data) * [1, -1]
                            temp_data = temp_data.reshape(-1,1,2)
                            #emp_data += [temp_data[0]]
                            data.append(temp_data)
                    else:
                        data = np.array(ori_data) * [1, -1]
                        data = data.reshape(-1,1,2)
                        #ata += [data[0]]
                yield data

    def rgba_to_rgb(self, hex):
        hex = hex.split("(")[1].split(")")[0].split(",")
        r = int(hex[0])
        g = int(hex[1])
        b = int(hex[2])
        return r, g, b

    def get_lable_color(self, label):
        for region in self.d_json['features']:
            if region['properties']['label_name'] == label:
                color1 = region['properties']['label_color']
                color = self.rgba_to_rgb(color1)
                return color

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
        if data_ds == roi_ds:
            return data
        else:
            return np.array(data * data_ds / roi_ds, dtype='int32')

    def get_json_label(self):
        """
        获取类别标签
        """
        json_list = []
        for region in self.d_json['features']:
            # json_list.append(list(region.values())[2]['label_name'])
            json_list.append(region['properties']['label_name'])
        json_label_list = sorted(list(set(json_list)))  # set() 函数创建一个无序不重复元素集
        return json_label_list

    def get_label_num(self):
        """
        获取各类别数量
        """
        label_num_dict = {}
        for label in self.get_json_label():
            label_num_dict[label] = 0

        for region in self.d_json['features']:
            label_num_dict[region['properties']['label_name']] += 1
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
            new_conts.append((np.array(cont) * level_ds / roi_level_ds).astype('int'))
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
    def __init__(self, save_dir, slide_path, data_dict: dict):
        """
        用于写json

        :param save_dir: 保存文件夹
        :param slide_name: 切片名
        :param data_dict: {
        # 切片级指标数据
        'data_indicators': {'指标1': {'value': 12.34, 'unit': '%'},
                            '指标2': {'value': 12.34, 'unit': '%'}},
        # 单层轮廓
        'cv': {'type': 'Polygon',
               'coordinates': conts,  # 模拟2个轮廓
               'color': (255, 255, 0),
               # 轮廓级指标数据
               'Polygon_indicators': [  # 注意指标个数应等于轮廓个数，指标结构不可更改
                                         {'指标1': {
                                             'value': 12.34,
                                             'unit': '%'},
                                             '指标2': {
                                                 'value': 12.34,
                                                 'unit': '%'}}] * len(conts),
               },
        # 内嵌轮廓
        'pa': {'type': 'Polygon',
               'coordinates': [conts, conts, conts],  # 模拟3个嵌套轮廓
               'color': (0, 255, 0),
               # 轮廓级指标数据
               'Polygon_indicators': [  # 注意指标个数应等于轮廓个数，指标结构不可更改
                                         {'指标1': {
                                             'value': 100,
                                             'unit': '%'},
                                             '指标2': {
                                                 'value': 200,
                                                 'unit': '%'}}] * len(conts) * 3,
               },
        # 点
        'po': {'type': 'Point',
               'coordinates': [[1, 3], [4, 2]],  # 模拟2个点
               'color': (0, 255, 0),
               # 轮廓级指标数据
               'Polygon_indicators': [  # 注意指标个数应等于轮廓个数，指标结构不可更改
                                         {'指标1': {
                                             'value': 1,
                                             'unit': '%'},
                                             '指标2': {
                                                 'value': 2,
                                                 'unit': '%'}}] * len([[1, 3], [4, 2]]),
               },
        # 线
        'li': {'type': 'Line',
               'coordinates': [[[1, 3], [4, 2]], [[1, 3], [4, 2]]],  # 模拟2条线
               'color': (0, 255, 0),
               # 轮廓级指标数据
               'Polygon_indicators': [  # 注意指标个数应等于轮廓个数，指标结构不可更改
                                         {'指标1': {
                                             'value': 10,
                                             'unit': '%'},
                                             '指标2': {
                                                 'value': 20,
                                                 'unit': '%'}}] * len([[1, 3], [4, 2]]),
               }
        """
        self.slide_path = slide_path
        self.slide_name, self.extension = os.path.splitext(os.path.basename(slide_path))
        self.save_path = os.path.join(save_dir, self.slide_name) + '.stjson'
        self.skeleton = {
            "type": "FeatureCollection",
            "project": {
                "name": self.slide_name + self.extension
            }
        }
        self.data_dict = data_dict

    def save_json(self, json_content, savepath):
        with open(savepath, 'w') as file:
            json.dump(json_content, file)

    def rgb_to_hex(self, color):
        r, g, b = color
        return ('{:02X}' * 3).format(r, g, b)

    def type_coord(self, type, ori_data):
        """
        按类型组织数据
        :param type: 'Point'、 'Line'、'Polygon'
        :param ori_data: 数据
        :return: 组织后数据
        """
        if type == 'Point':
            data = (np.array(ori_data) * [1, -1]).tolist()
        elif type == 'Line':
            data = (np.array(ori_data) * [1, -1]).tolist()
        elif type == 'Polygon':
            if isinstance(ori_data, list):  # 是list的话判断为内嵌结构
                data = []
                for temp_data in ori_data:
                    temp_data = (np.array(temp_data).reshape(-1, 2) * [1, -1]).tolist()
                    temp_data += [temp_data[0]]
                    data.append(temp_data)
            else:
                data = (np.array(ori_data).reshape(-1, 2) * [1, -1]).tolist()
                data += [data[0]]
                data = [data]
        return data

    def fill_json(self):
        # 保存流程计算指标
        if 'data_indicators' in self.data_dict:
            self.skeleton['data_indicators'] = self.data_dict.pop('data_indicators')
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        id = 1
        features = []
        for label, geometry_dict in self.data_dict.items():
            type = geometry_dict['type'] if 'type' in geometry_dict else None
            # color = self.rgb_to_hex(geometry_dict['color']) if 'color' in geometry_dict else self.rgb_to_hex((0, 0, 0))
            color = str(geometry_dict['color'])
            if 'coordinates' in geometry_dict:
                coordinates = geometry_dict['coordinates']
                polygon_indicators = geometry_dict[
                    'Polygon_indicators'] if 'Polygon_indicators' in geometry_dict else None

                for i, cont in enumerate(coordinates):
                    coord = self.type_coord(type, cont)
                    indicators = polygon_indicators[i] if polygon_indicators else {}
                    features.append({"type": "Feature",
                                     "geometry": {
                                         "type": type,
                                         "coordinates": coord
                                     },
                                     "properties": {
                                         "annotation_source_type": "AI",
                                         "create_time": time,
                                         "label_name": label,
                                         "label_color": color,
                                         "data_indicators": indicators
                                     },
                                     'id': 'ai'+str(label)+str(id)+str(random.randint(10, 99))
                                     })
                    id += 1
        self.skeleton['features'] = features

    def main(self):
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
        self.read_svs = TiffImg(svs_path)
        self.save_path = os.path.join(save_dir, self.read_svs.slide_name)


    def translucence(self, img_rgb_ds, conts, color):
        mask_2 = img_rgb_ds.copy()
        for cont in conts:
            cv2.drawContours(mask_2, [cont], -1, color, -1)
        result_img = cv2.addWeighted(img_rgb_ds, 0.78, mask_2, 0.22, 1)
        return result_img


    def single_add(self, level_ds , thickness, translucence_label_list=None,
                   save_svs=False):
        """
        单张绘制展示图

        :param level_ds: 缩放比例
        :param thickness: 绘图线宽

        :return:
        """
        if translucence_label_list is None:
            translucence_label_list = []
        print('read img start')
        img_ds = self.read_svs.get_img_ds(level_ds)
        print('read img end')
        json_label = self.read_json.get_json_label()
        print(json_label)

        for i, label in enumerate(json_label):
            color =self.read_json.get_lable_color(label)
            conts_ds1 = self.read_json.get_conts_list(label=label)
            print(conts_ds1)
            conts = self.read_json.transform_conts_level(conts_ds1, 1, roi_level_ds=level_ds)

            for cont in conts:
                x, y, w, h = cv2.boundingRect(cont)
                # print(x,y,w,h)
                cv2.putText(img_ds, str(label), tuple([int(x), int(y)]),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, [0, 0, 0], 3)

            if label in translucence_label_list:
                img_ds = self.translucence(img_ds, conts, color)
            else:
                cv2.drawContours(img_ds, conts, -1, color, thickness)

        if save_svs:
            array_to_STAI(img_ds, self.save_path + 'modify.svs')
        else:
            img_rgb_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.save_path + '.png', img_rgb_ds)

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
            if cv2.contourArea(cont) < 2:  #轮廓面积
                continue
            M = cv2.moments(cont)   #轮廓的矩
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            cv2.circle(img, tuple([int(center_x), int(center_y)]), 2, (255, 0, 0), 4)
            cv2.putText(img, str((center_x, img_h - center_y)), tuple([x, y]), font, 1, (255, 0, 0), 1)
            str_1 += str((center_x, img_h - center_y)) + '\n'
        if name:
            with open(os.path.join(self.save_path, name) + '.txt', "w") as f:
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

def draw_test():
    mask = np.zeros((200, 200), dtype='uint8')
    cv2.rectangle(mask, (20, 20), (50, 50), (255, 255, 255), thickness=-1)
    cv2.circle(mask, (100, 100), 50, (255, 255, 255), -1)
    # cv2.imshow('mask',mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


if __name__ == '__main__':
    pass