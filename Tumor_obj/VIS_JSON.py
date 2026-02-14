#encoding:utf-8
import json
import os

# os.add_dll_directory(r'D:/Anaconda3/Library/openslide-win64-20221217/bin')
import cv2
import numpy as np
import glob
import tifffile
import openslide


def array_to_STAI(img, end_path, page_amount=4, sampel=2):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with tifffile.TiffWriter(end_path, bigtiff=True, append=False) as tif_new:
        tif_new.save(img,
                     photometric='rgb',
                     # compress='jpeg',
                     planarconfig='CONTIG',
                     tile=(256, 256),
                     subsampling=(1, 1),
                     subfiletype=9,
                     )
        for i in range(page_amount):
            img = cv2.resize(img, (0, 0), fx=1 / sampel, fy=1 / sampel)
            tif_new.save(img,
                         photometric='rgb',
                         # compress='jpeg',
                         planarconfig='CONTIG',
                         tile=(256, 256),
                         subsampling=(1, 1),
                         subfiletype=9,
                         )


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
        # print('self.slide.get_thumbnail((w_ds, h_ds))',np.array(self.slide.get_thumbnail((w_ds, h_ds))).shape)# (11424, 14623, 3)
        img_ds = np.array(self.slide.get_thumbnail((w_ds, h_ds)))[:, :, :3]  # 切片似乎没必要
        # plt.imshow(img_ds)
        # plt.show()
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


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def do_vis(svs_path, level=8):
    name = svs_path.split('/')[-1].split('.')[0]
    j_path = os.path.join(r'/home/zyj/demo/xianpao/data/test/my_json', '{}.json'.format(name))
    data = read_json(j_path)  # 修改json路径

    read_svs = OpenSlideImg(svs_path)
    img_ds = read_svs.get_img_ds(level_ds=level)
    img_rgb_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)

    for feature in data['features']:
        contour_coordinates = feature['geometry']['coordinates'][0]

        # 将轮廓坐标转换成整数类型
        contour_coordinates = np.array(contour_coordinates, dtype=np.int32) // level

        cv2.drawContours(img_rgb_ds, [contour_coordinates], -1, (0, 255, 0), thickness=2)

    array_to_STAI(img_rgb_ds, os.path.join(r'/home/zyj/demo/xianpao/data/test/vis', '{}.svs'.format(name)))



if __name__ == '__main__':
    all_svs_path = glob.glob(r'/home/zyj/demo/xianpao/data/test/ST20Mm-LG-MD-ML-PG-SD-424-1-000030_001.svs')  # svs路径
    # all_svs_path = glob.glob(r"D:\program\src\pre_data\test_klghs_ts\ST20Mf-LG-MD-ML-PG-SD-422-1-000031_001.svs")
    # print(all_svs_path)
    print(len(all_svs_path))
    for svs_path in all_svs_path:
        print(svs_path)
        do_vis(svs_path)

#     do_vis('/home/fhs/data-fhs/SDEV/PG/STANDARD_TESTSET/svs/ST18Rf-LG-MD-ML-PG-SD-317-1-000019.svs')
#     do_vis('/home/PBdata_mask_ori/Whole-tissue/ST16R-WT-282/ST16Rf-LG-MD-ML-PG-SD-282-4-000119-4F.svs')
#     do_vis('./PINGSHEN_SLIDE_PART/ST20Rm-LG-MD-ML-PG-SD-321-1-000014_7543_7686.svs')
#     do_vis('./PINGSHEN_SLIDE_COPY/ST16Rf-LG-MD-ML-PG-SD-282-1-000023-1F.svs')
