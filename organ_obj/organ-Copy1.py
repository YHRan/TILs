import os
from skimage import filters
import cv2
import numpy as np
from skimage.morphology import disk
from read_img import ImgFunc ,array_to_STAI
from read_image import OpenSlideImg, TiffImg
from find_roi import *
# from st_json import WriteJson
script_dir = os.path.dirname(os.path.abspath(__file__))

class Coord_tool():
    def __init__(self):
        pass

    @classmethod
    def coord2xywh(cls, coord):
        w, h = coord[1] - coord[0]
        x, y = coord[0]
        return [x, y, w, h]

    @classmethod
    def xywh2coord(cls, x, y, w, h):
        return [[x, y], [x + w, y + h]]

    @classmethod
    def xywh2xyxy(cls, xywh):
        x, y, w, h = xywh
        return [x, y, x + w, y + h]

    @classmethod
    def xywh2angular_point(cls, x, y, w, h):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    @classmethod
    def get_all_file(cls, ori_path, postfix=''):
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

    @classmethod
    def transform_conts_level(cls, conts, conts_level_ds, roi_level_ds=1):
        """
        轮廓修改缩放比例

        :param conts: 轮廓
        :param conts_level_ds: 轮廓的缩放比例
        :param roi_level_ds: 目标缩放比例
        """
        return [np.array(cont * conts_level_ds / roi_level_ds, dtype='int') for cont in conts]

class FindRoiOophoron(FindRoi):

    def __init__(self, svs_read: OpenSlideImg or TiffImg, img_ds32):
        super(FindRoiOophoron, self).__init__()
        self.svs_read = svs_read
        self.img_ds32 = img_ds32
        self.img_read = ImgFunc()

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
        # cv2.imwrite('gray_save.png', gray_save)
        gray = (filters.roberts(gray_save) * 255).astype('uint8')
        # cv2.imwrite('gray.png', gray)
        ret3, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU二值化
        k_close = np.ones((3, 3), np.uint8)
        close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=3)
        # cv2.imwrite('close.png', close)
        _,mask_cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_xywhs = []
        for cnt in mask_cnts:
            if cv2.contourArea(cnt) > 100:
                xywh = cv2.boundingRect(cnt)
                angular_point = Coord_tool.xywh2angular_point(*xywh)
                # num = 0
                # for point in angular_point:
                #     if not point_in_edge(point, img_shape=self.img_ds32.shape[:2], edge_rate=cut_edge_rate):
                #         num += 1
                # if num == 4:
                roi_xywhs.append(xywh)

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
        _,cnts, _ = cv2.findContours(mask_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        conts_ds32 = [cont for cont in cnts if cv2.contourArea(cont) > 20000]
        return conts_ds32

if __name__ == '__main__':
    svs_path = r'/home/lyb/data_yhr/03-yhr/External_project/TILs_draw_point/svs_260209_'
    for svs_data in os.listdir(svs_path):
        if svs_data.endswith('.svs'):
            print('svs_data_process:', svs_data)
            svs_read = OpenSlideImg(os.path.join(svs_path,svs_data))
            svs_read1 = TiffImg(os.path.join(svs_path,svs_data))
            # find roi organ
            image_ds1 = svs_read1.get_img_ds(1)
            image_ds4 = svs_read.get_img_ds(level_ds=4)
            mask = np.zeros_like(image_ds1)
            print('image_ds4.shape:', image_ds4.shape)
            image_ds16 = svs_read.get_img_ds(level_ds=4)
            print('image_ds16.shape:', image_ds16.shape)
            find_roi = FindRoiOophoron(svs_read, image_ds16)
            roi_xywhs = find_roi.pretreatment(cut_edge_rate=0.05)
            out_conts_ds16 = find_roi.get_out_conts(roi_xywhs)
            print(len(out_conts_ds16))
            all_conts_organ = [np.array(cnt) for cnt in out_conts_ds16 ] #d4 *4  ds2 *8
            trans_ds1_cont = [np.array(cnt * 4) for cnt in out_conts_ds16 ]
            cv2.drawContours(image_ds4, all_conts_organ , -1, (255, 255, 0), 10)
            cv2.drawContours(mask, trans_ds1_cont , -1, (255, 255, 255), -1)
            # save_path = os.path.join(script_dir, 'result_test4')
            save_path = r'/home/lyb/data_yhr/03-yhr/External_project/TILs_draw_point/svs_260209_/RES'
            os.makedirs(save_path, exist_ok=True)

            # trans_cont = [np.array(cnt * 8) for cnt in out_conts_ds16]
            # json_dict = {}
            # json_dict['organ'] = trans_cont
            # WriteJson(save_path, os.path.join(svs_path,svs_data), json_dict).main()

            cv2.imwrite(os.path.join(save_path, svs_data.rpartition('.')[0]) + '_organ.png', mask)
            img_rgb_ds1 = cv2.cvtColor(image_ds4, cv2.COLOR_BGR2RGB)
            array_to_STAI(img_rgb_ds1, os.path.join(save_path, svs_read.slide_name) + 'organ.svs')
            print('{}: have done'.format(svs_data))