import os
import cv2
import itertools
import glob
import numpy as np
from random import randint
import argparse
from st_json import ReadJson, match_svs_json, get_all_file
from read_img import ImgFunc, OpenSlideImg  # 读图函数
from utils import CoordTool
import yaml


class MakeData():
    def __init__(self, svs_path, json_path, save_path, data_flag):
        self.read_img = ImgFunc()
        self.read_json = ReadJson(json_path)
        self.read_svs = OpenSlideImg(svs_path)
        self.save_path = save_path
        self.data_flag = data_flag
        self.img_ds = self.read_svs.get_img_ds(data_flag['data_info']['level_ds'])

    def _get_patch_list(self, label_scope):
        """
        滑窗切图法，返回的coord_list存储了每个patch左上角和右下角的坐标: [[(x_min,y_min),(x_max,y_max)],[],...]
        """
        step_size = self.data_flag['data_info']['step']
        y_size, x_size = self.data_flag['data_info']['img_h_w']
        coord_list = []

        # 防止没有标注区域时索引报错
        if not label_scope:
            return coord_list

        x_min = label_scope[0][0]
        y_min = label_scope[0][1]
        x_max = label_scope[1][0]
        y_max = label_scope[1][1]
        for x in range(x_min, x_max, step_size):
            if x + step_size >= x_max:
                x = x_max - step_size
            for y in range(y_min, y_max, step_size):
                if y + step_size >= y_max:
                    y = y_max - step_size
                coord_list.append([(int(x), int(y)),
                                   (int(x + x_size), int(y + y_size))])
        return coord_list

    def get_label_scope(self):
        """
        获取标注区域（为了避免切分出许多不存在标注的patch，首先定位有标注区域的范围:(左上角坐标，右下角坐标)？）
        :return:
        """
        all_boxs = []
        for label in self.data_flag['data_info']['label_list']:
            boxs = self.read_json.get_boxs(label)
            all_boxs += list(boxs)
        if all_boxs == []:
            return []

        all_points = np.array(all_boxs).reshape(-1, 2)
        point_min = np.min(all_points, axis=0)
        point_max = np.max(all_points, axis=0)
        level_ds = self.data_flag['data_info']['level_ds']
        img_h, img_w = self.data_flag['data_info']['img_h_w']
        max_h, max_w = self.read_svs.get_img_ds_shape(level_ds)  # 缩略图shape
        x_min, y_min = np.array(point_min / level_ds, dtype='int')
        x_max, y_max = np.array(point_max / level_ds, dtype='int')
        # print(x_min, y_min,x_max, y_max)
        if x_max % img_w != 0:
            mod_x = x_max // img_w
            if (mod_x + 1) * img_w <= max_w:
                x_max = (mod_x + 1) * img_w
            else:
                x_max = mod_x * img_w

        if y_max % img_h != 0:
            mod_y = y_max // img_h
            if (mod_y + 1) * img_h <= max_h:
                y_max = (mod_y + 1) * img_h
            else:
                y_max = mod_y * img_h

        return [[x_min, y_min], [x_max, y_max]]  # X_MIN+512=X_MAX, Y_MIN+512=Y_MAX

    def boxs_in_coord(self, coord, boxs):
        """
        获取框在小图内坐标
        :param coord: 小图在整张上的
        :param boxs: 全部标注框
        :return:
        """

        def box2points(box):
            point_1 = box[0]
            point_2 = [box[1][0], box[0][1]]
            point_3 = box[1]
            point_4 = [box[0][0], box[1][1]]
            return [point_1, point_2, point_3, point_4]

        def point_in_coord(point, coord):
            if coord[0][0] < point[0] < coord[1][0]:
                if coord[0][1] < point[1] < coord[1][1]:
                    return True
                else:
                    return False
            else:
                return False

        boxs_in_coord = []
        for box in boxs:
            point_list = box2points(box)
            for point in point_list:
                if point_in_coord(point, coord):
                    boxs_in_coord.append(np.array(box) - coord[0])
                    break
        boxs_in_coord = np.array(boxs_in_coord)
        boxs_in_coord[boxs_in_coord < 0] = 0
        h, w = self.data_flag['data_info']['img_h_w']
        for i, box in enumerate(boxs_in_coord):
            for j, point in enumerate(box):
                if point[0] > w:
                    boxs_in_coord[i][j][0] = w
                if point[1] > h:
                    boxs_in_coord[i][j][1] = h
        return boxs_in_coord

    def code_data(self, label, boxs_):
        """
        组织数据格式
        :param label: 数据类别编码
        :param boxs_: 数据框
        :return:
        """
        if self.data_flag['model_type'] == 'yolo_v5':
            data = ''
            for box in boxs_:
                x, y = box[0]
                w, h = box[1] - box[0]
                img_h, img_w = self.data_flag['data_info']['img_h_w']
                str_ = str(label) + ' ' + str(int(x + w / 2) / img_w) + ' ' + str(int(
                    y + h / 2) / img_h) + ' ' + str(int(w) / img_w) + ' ' + str(int(h) / img_h) + '\n'
                data += str_

        elif self.data_flag['model_type'] == 'yolo_v3':
            data = ''
            for box in boxs_:
                str_ = str(int(box[0][0])) + '	' + str(int(box[0][1])) + '	' + str(int(box[1][0])) + '	' + str(
                    int(box[1][1])) + '	' + str(label) + '\n'
                data += str_
        else:
            data = ''
        return data

    # Todo: 缩放切图测试未通过。待处理
    # def cut_img(self, coord):
    #     level_ds = self.data_flag['data_info']['level_ds']
    #     if level_ds == 1:
    #         img = self.read_svs.get_coord_img(coord)
    #     else:
    #         coord = np.array(coord) * level_ds
    #         img = self.read_svs.get_coord_img(coord)
    #         img = cv2.resize(img, (0, 0), fx=1 / level_ds, fy=1 / level_ds, interpolation=cv2.INTER_LANCZOS4)
    #     return img

    def cut_img(self, coord):
        img = self.read_img.get_patch_img(self.img_ds, coord)
        return img

    def yolo_data(self):
        label_boxs_dict = {}
        for label in self.data_flag['data_info']['label_list']:
            label_boxs_dict[label] = []

        label_code_dict = {}
        for i, label in enumerate(self.data_flag['data_info']['label_list']):
            boxs = self.read_json.get_boxs(label)
            boxs = self.read_json.change_ds(boxs, 1, self.data_flag['data_info']['level_ds'])
            label_boxs_dict[label] = boxs
            label_code_dict[label] = i
        print('label code', label_code_dict)

        # label_scope = self.get_label_scope()
        # if label_scope == []:
        #     print('None')
        #     return None

        h, w = self.read_svs.get_img_ds_shape(self.data_flag['data_info']['level_ds'])
        label_scope = [[0, 0], [w, h]]
        coord_list = self._get_patch_list(label_scope)
        # print('patch num:', len(coord_list))
        patch_num = 0
        for coord in coord_list:
            str_ = ''
            for label, boxs in label_boxs_dict.items():
                boxs_in_img = self.boxs_in_coord(coord, boxs)

                # img = self.cut_img(coord)
                # for box in boxs_in_img:
                #     cv2.rectangle(img, tuple(box[0]),tuple(box[1]), (0,255,0), 2)
                # cv2.imshow('img',img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                if boxs_in_img == []:
                    break

                str_ += self.code_data(label_code_dict[label], boxs_in_img)
            if str_ == "":
                continue

            with open(self.save_path + '/' + self.read_svs.slide_name + "_" + str(coord[0]) + '.txt', 'w'):
                pass

            img = self.cut_img(coord)
            patch_num += 1
            with open(self.save_path + '/' + self.read_svs.slide_name + "_" + str(coord[0]) + '.txt', 'a') as file:
                file.write(str_)
            cv2.imwrite(self.save_path + '/' + self.read_svs.slide_name + "_" + str(coord[0]) + '.png', img)
        print('patch num:', patch_num)

    def conts_in_coord(self, coord, conts):
        def cont2points(cont):
            x, y, w, h = cv2.boundingRect(cont)
            mask = np.zeros((h, w))
            cv2.drawContours(mask, [cont - [x, y]], -1, 255, -1)

            points = []
            for _ in range(300):
                point_y = randint(0, h - 1)
                point_x = randint(0, w - 1)
                if mask[point_y, point_x] == 255:
                    point = np.array([point_x, point_y])
                    points.append(point)
                    # break
                else:
                    point = []

            # point_1 = [x, y]
            # point_2 = [x + w, y]
            # point_3 = [x + w, y + h]
            # point_4 = [x, y + h]
            # return [point_1, point_2, point_3, point_4]

            return np.array(points) + np.array(coord[0])

        def point_in_coord(point, coord):
            if coord[0][0] < point[0] < coord[1][0]:
                if coord[0][1] < point[1] < coord[1][1]:
                    return True
            return False

        conts_in_coord = []
        for cont in conts:
            point_list = cont2points(cont)
            for point in point_list:
                if point_in_coord(point, coord):
                    conts_in_coord.append(np.array(cont) - coord[0])
                    break
        conts_in_coord = np.array(conts_in_coord)
        # boxs_in_coord[boxs_in_coord < 0] = 0
        return conts_in_coord

    def u2net_data(self):
        label_conts_dict = {}
        for label in self.data_flag['data_info']['label_list']:
            label_conts_dict[label] = []

        label_code_dict = {}
        for i, label in enumerate(self.data_flag['data_info']['label_list']):
            conts = self.read_json.get_conts_list(label)
            conts = self.read_json.transform_conts_level(conts, 1, self.data_flag['data_info']['level_ds'])
            label_conts_dict[label] = conts
            label_code_dict[label] = i
        #         print('label_conts_dict',label_conts_dict.keys(),len(label_conts_dict['G']))# dict_keys(['G']) 1954 ,  所有目标(G)的轮廓坐标集合(np.array格式)
        # print('label code', label_code_dict)# {'G':0}
        label_scope = self.get_label_scope()
        # print('label_scope',label_scope)# [[10280, 5377], [19968, 16896]]
        coord_list = self._get_patch_list(label_scope)  # 所有切割得到的patch的(左上角，右下角)坐标list

        mask_all = []
        for label, conts in label_conts_dict.items():
            mask = np.zeros(self.read_svs.get_img_ds_shape(self.data_flag['data_info']['level_ds']))
            cv2.drawContours(mask, conts, -1, 255, -1)
            mask_all.append(mask)

        img_h, img_w = self.data_flag['data_info']['img_h_w']
        img_area = img_h * img_w
        mask_all = np.array(mask_all)
        print('mask_all', mask_all.shape)  # (1, img_h, img_w) mask_all 的shape和对应的rgb的shape一致(都是完整大图的shape) ， 1是因为只有一个类G
        for coord in coord_list:
            # print('coord: ',coord)
            mask_patch = mask_all[:, coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]  # 将大的mask切分成和rgb切分结果对应的小patch

            print('mask_patch', mask_patch.shape)  # (1, 512, 512)
            if np.count_nonzero(mask_patch) > 0.001 * img_area:  # 各类别占比小于0.1% 就不要了
                # print('keep')
                for i, label_mask_patch in enumerate(mask_patch):
                    if not os.path.exists(os.path.join(self.save_path, str(i))):
                        os.makedirs(os.path.join(self.save_path, str(i)))
                    mask_name = self.read_svs.slide_name + "_" + str(coord[0])
                    cv2.imwrite(os.path.join(self.save_path, str(i), mask_name) + '.png', label_mask_patch)

                img = self.cut_img(coord)
                if not os.path.exists(os.path.join(self.save_path, 'img-bowl')):
                    os.makedirs(os.path.join(self.save_path, 'img-bowl'))
                img_name = self.read_svs.slide_name + "_" + str(coord[0])
                cv2.imwrite(os.path.join(self.save_path, 'img-bowl', img_name) + '.png', img)

    def marsk_rcnn_data(self):
        label_conts_dict = {}
        for label in self.data_flag['data_info']['label_list']:
            label_conts_dict[label] = []

        label_code_dict = {}
        for i, label in enumerate(self.data_flag['data_info']['label_list']):
            conts = self.read_json.get_conts_list(label)
            conts = self.read_json.transform_conts_level(conts, 1, self.data_flag['data_info']['level_ds'])
            label_conts_dict[label] = conts
            label_code_dict[label] = i
        print('label code', label_code_dict)
        label_scope = self.get_label_scope()
        coord_list = self._get_patch_list(label_scope)
        for coord in coord_list:
            save_dir = os.path.join(self.save_path, self.read_svs.slide_name + "_" + str(coord[0]))
            os.mkdir(save_dir)
            for label, conts in label_conts_dict.items():
                conts_in_img = self.conts_in_coord(coord, conts)
                if conts_in_img == []:
                    break
                for i, cont in enumerate(conts_in_img):
                    mask = np.zeros(self.data_flag['data_info']['img_h_w'])
                    cv2.drawContours(mask, [cont], -1, 255, -1)
                    cv2.imwrite(os.path.join(save_dir, self.read_svs.slide_name + '_mask_' + str(
                        label_code_dict[label]) + '_' + str(i)) + '.png', mask)
            img = self.cut_img(coord)
            cv2.imwrite(save_dir + '/' + self.read_svs.slide_name + "_" + str(coord[0]) + '.png', img)

    def cont_to_coord(self, cont):
        x, y, w, h = cv2.boundingRect(cont)
        padding_rate = self.data_flag['data_info']['padding_rate']
        if padding_rate > 0:
            w = int(w * (1 + 2 * padding_rate))
            h = int(h * (1 + 2 * padding_rate))
            x -= int(w * padding_rate)
            y -= int(h * padding_rate)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            new_coord = CoordTool.xywh2coord(x, y, w, h)
        else:
            new_coord = CoordTool.xywh2coord(x, y, w, h)
        return new_coord

    def cls_data(self):
        label_coords_dict = {}
        for label in self.data_flag['data_info']['label_list']:
            label_save_path = os.path.join(self.save_path, label)
            if not os.path.exists(label_save_path):
                os.makedirs(label_save_path)
            label_coords_dict[label] = []

        label_num_dict = {}
        for i, label in enumerate(self.data_flag['data_info']['label_list']):
            conts = self.read_json.get_conts_list(label)
            conts = self.read_json.transform_conts_level(conts, 1, self.data_flag['data_info']['level_ds'])
            coords = []
            for cont in conts:
                coord = self.cont_to_coord(cont)
                coords.append(coord)
            label_coords_dict[label] = coords
            label_num_dict[label] = len(coords)

        print('label num', label_num_dict)

        for label, coords in label_coords_dict.items():
            label_save_path = os.path.join(self.save_path, label)
            for coord in coords:
                img = self.cut_img(coord)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(label_save_path + '/' + self.read_svs.slide_name + "_" + str(coord[0]) + '.png', img_rgb)

    def main(self):
        if self.data_flag['data_info']['label_list'] == []:
            self.data_flag['data_info']['label_list'] = self.read_json.get_json_label()
        if self.data_flag['model_type'] == 'yolo_v5' or self.data_flag['model_type'] == 'yolo_v3':
            self.yolo_data()
        elif self.data_flag['model_type'] == 'u2net':
            self.u2net_data()
        elif self.data_flag['model_type'] == 'mask_rcnn':
            self.marsk_rcnn_data()
        elif self.data_flag['model_type'] == 'cls':
            self.cls_data()


def get_hyp_data(hyp):
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    json_dir = hyp.get('json_dir')
    svs_dir = hyp.get('svs_dir')
    save_dir = hyp.get('save_dir')

    model_type = hyp.get('data_flag')['model_type']
    label_list = hyp.get('data_flag')['label_list']
    level_ds = hyp.get('data_flag')['level_ds']

    if model_type == 'cls':
        padding_rate = hyp.get('data_flag')['padding_rate']
        data_flag = {
            'model_type': model_type,  # 数据集类型
            'data_info': {  # 数据细节
                'label_list': label_list,  # 类别列表。空则处理全部标签。
                'level_ds': level_ds,  # 缩放比例。
                'padding_rate': padding_rate}  # 边缘补充
        }
        return json_dir, svs_dir, save_dir, data_flag
    else:
        img_h_w = hyp.get('data_flag')['img_h_w']
        step = hyp.get('data_flag')['step']
        data_flag = {
            'model_type': model_type,  # 数据集类型 yolo_v3
            'data_info': {  # 数据细节
                'label_list': label_list,  # 类别列表。 空则处理全部标签。
                'level_ds': level_ds,  # 缩放比例。
                'img_h_w': img_h_w,  # 图片高、宽。
                'step': step}  # 切割步长。
        }
        return json_dir, svs_dir, save_dir, data_flag


def make_data(hyp):
    json_dir, svs_dir, save_dir, data_flag = get_hyp_data(hyp)
    json_list = get_all_file(json_dir, '.json')
    svs_list = get_all_file(svs_dir, '.svs')
    svs_list, json_list = match_svs_json(svs_list, json_list)
    # svs_list=glob.glob(r'D:\ProjectSpace\STRUCTURE_DEV\PG\prepare_dataset\dataset\svs\*.svs')
    # json_list=glob.glob(r'D:\ProjectSpace\STRUCTURE_DEV\PG\dataset_prepare_train\cell\json_ours\*.json')
    for svs_path, json_path in zip(svs_list, json_list):
        print(svs_path)
        MakeData(svs_path, json_path, save_dir, data_flag=data_flag).main()


parser = argparse.ArgumentParser()
parser.add_argument('--hyp_path', default=r'./make_data.yaml', type=str)
args = parser.parse_args()
print(args.hyp_path)
make_data(args.hyp_path)
