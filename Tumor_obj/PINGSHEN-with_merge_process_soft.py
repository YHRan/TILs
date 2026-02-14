import glob
import os
# os.add_dll_directory(r'D:/Anaconda3/Library/openslide-win64-20221217/bin')
import time
import copy
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from config import CFG
from dataset import BuildDataset
from net import net
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import load_img, load_msk, plot_batch
from config import CFG

from st_json import WriteJson


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


class BuildDataset(Dataset):
    def __init__(self, imgs_list, transforms=None):
        self.imgs_list = imgs_list
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        image = self.imgs_list[idx]  # uint8

        label = np.zeros(image.shape)

        sample = {'image': image, 'label': label}

        if self.transforms:
            sample = self.transforms(sample)

        return sample['image']


class MedullaClassify():
    def __init__(self, pt_model, preprocess):
        self.model = pt_model
        self.preprocess = preprocess

    def run(self, raw_img, thres=0.9):
        split_start = time.time()
        raw_imgs, cut_start_coords, cut_row_num, cut_col_num, cut_size, cut_stride, height_padding, width_padding = self._split_img(
            raw_img)
        split_end = time.time()
        print('切图耗时：', split_end - split_start)
        infer_dataset = BuildDataset(raw_imgs, self.preprocess)
        infer_loader = DataLoader(infer_dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=True,
                                  drop_last=False)
        print('infer_loader:', len(infer_loader))

        all_mask = []

        start_time = time.time()

        for img_batch in tqdm(infer_loader):
            img_batch = img_batch.to(CFG.device).float()
            ###################前向推理逻辑#####################
            if CFG.model_type == 'u2net_p' or CFG.model_type == 'u2net':
                batch_output, _, _, _, _, _, _ = self.model(img_batch)
            elif CFG.model_type in ['smp', 'mobile_sam', 'cenet']:
                batch_output = self.model(img_batch)
                batch_output = torch.sigmoid(batch_output)
            #             binary_mask = ((batch_output > thres) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)# bs,512,512,1
            binary_mask = batch_output.permute(0, 2, 3, 1).detach().cpu().numpy()  # bs,512,512,1
            ###################################################
            all_mask.extend(binary_mask)

        end_time = time.time()
        print((end_time - start_time) / len(raw_imgs))  # 0.06

        mask = self._build_MASK(height_padding, width_padding, all_mask, cut_start_coords, cut_size, cut_stride, thres)

        all_conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return all_conts

    def _split_img(self, image_array):
        image_array = cv2.copyMakeBorder(image_array, 0, 640, 0, 640, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        height, width = image_array.shape[0], image_array.shape[1]  # padding后的高和宽

        cut_size = 640
        cut_stride = 480
        cut_row_num = (height - cut_size) // cut_stride + 1
        cut_col_num = (width - cut_size) // cut_stride + 1
        image_num = cut_row_num * cut_col_num
        small_imgs = np.zeros((image_num, cut_size, cut_size, 3), dtype=np.uint8)
        num = 0
        i, j = 0, 0

        cut_start_coords = []  # 记录每个patch相对于image_array的左上角坐标

        while i <= image_array.shape[0] - cut_size:
            while j <= image_array.shape[1] - cut_size:
                subimage = image_array[i:i + cut_size, j:j + cut_size, :]
                small_imgs[num] = subimage
                cut_start_coords.append([i, j])  # 记录当前切分patch的左上角坐标
                j = j + cut_stride
                num += 1
            j = 0
            i = i + cut_stride

        return small_imgs, cut_start_coords, cut_row_num, cut_col_num, cut_size, cut_stride, height, width

    def _build_MASK(self, height_padding, width_padding, all_mask, cut_start_coords, cut_size, cut_stride, threshold):
        canvas = np.zeros((height_padding, width_padding), dtype=np.float32)
        counts = np.ones((height_padding, width_padding), dtype=np.uint8)

        overlap_size = cut_size - cut_stride

        for idx, (start_row, start_col) in enumerate(cut_start_coords):
            end_row = start_row + cut_size
            end_col = start_col + cut_size

            mask = all_mask[idx]

            if CFG.valid_img_size != 640 and CFG.PATCH_SIZE == 640:
                mask = cv2.resize(mask, (640, 640))
                mask = mask[:, :, np.newaxis]

            # 在重叠区域进行累加
            canvas[start_row:end_row, start_col:end_col] += mask[:, :, 0]
            counts[start_row:(start_row + overlap_size), start_col:(start_col + overlap_size)] += 1

        # 最终结果是累加平均
        canvas /= counts

        # 二值化
        binary_mask = ((canvas > threshold) * 255).astype(np.uint8)

        return binary_mask


def convert_int32_to_int(data):
    if isinstance(data, dict):
        return {k: convert_int32_to_int(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_int32_to_int(item) for item in data]
    elif isinstance(data, np.int32) or isinstance(data, np.int64):
        return int(data)
    else:
        return data


# 检查一个轮廓的中心点是否在 ROI 轮廓区域内
def is_contour_center_inside_roi(contour, roi_contour):
    centroid = np.mean(contour, axis=0)
    x, y = centroid[0]
    centroid = [x, y]

    return cv2.pointPolygonTest(roi_contour, tuple(centroid), False) >= 0


if __name__ == '__main__':
    IS_REVIEW = False
    level = 1
    label_name = 'xianpao'
#     all_svs_path = glob.glob(r"/home/zyj/SD/xianpao/data/test/*.svs")
    all_svs_path = glob.glob(r"/home/zyj/MD/keliguanhongse/data/val/*.svs")
    print(len(all_svs_path))
    save_dir = r'/home/zyj/SD/xianpao/infer_test_fu'
    model_path = r'/home/zyj/SD/xianpao/exp/analys_up_json_fu/save_models/u2net_p_best_epoch-398_best_iter-13134_val_loss-1.5022693077723186_best_dice-0.9634643197059631_best_jaccard-0.9296596646308899_zyj.bin'

    net.load_state_dict(torch.load(model_path))
    net.to(CFG.device)
    net.eval()
   

    preprocess = CFG.data_transforms(CFG.train_img_size_, CFG.valid_img_size)['valid']

    for svs_path in all_svs_path:
        print(svs_path)
        if IS_REVIEW:
            save_name_ = svs_path.split(os.sep)[-1].split('.')[0]  # 带着x_min,y_min后缀的
            save_name = save_name_.split('.')[0].split('_')[0]  # 保存json时去掉后缀
            print('save name: ', save_name)
        else:
            save_name = svs_path.split(os.sep)[-1].split('.')[0]

        read_slide_start = time.time()
        svs_read = OpenSlideImg(svs_path)
        img_ds4 = svs_read.get_img_ds(level_ds=level)
        read_slide_end = time.time()
        print('读切片耗时：', read_slide_end - read_slide_start)

        all_conts = MedullaClassify(net, preprocess).run(img_ds4)

        # 过滤小面积轮廓
        area_threshold = 500
        all_conts = [contour for contour in all_conts if cv2.contourArea(contour) > area_threshold]
        num_ori = len(all_conts)
        print(num_ori)

        # 转换到原图坐标
        if IS_REVIEW:
            s = time.time()
            filtered_conts = []
            for cell_cont in all_conts:
                cell_cont *= level  # 提前转换到ds1
                x_min, y_min = save_name_.split('_')[-2:]
                offset = np.array([int(x_min), int(y_min)])  # ds1的
                cell_cont = cell_cont + offset
                filtered_conts.append(cell_cont)
            print('转换坐标时间={}'.format(time.time() - s))

            all_conts = filtered_conts
        else:
            all_conts = [cont * level for cont in all_conts]  # 提前转换到ds1

        num_roi_filtered = len(all_conts)
        print(num_roi_filtered)

        print('--------------------------summary--------------------------')
        print(f'\n初始轮廓数量={num_ori}\nROI区域过滤后数量={num_roi_filtered}')

        #################################################### 使用st_json保存成旧json格式####################################################

        WriteJson(os.path.join(save_dir, 'old_json', save_name + '.json'), svs_path, {label_name: all_conts}).main()
        #################################################### 使用st_json保存成旧json格式####################################################

        # 保存成新json格式
        to_save = {}
        features = []

        for cont in all_conts:

            # cont:Nx1x2

            cur_cont = []
            for xy in cont:
                cur_cont.append(list(xy[0]))

            coors = copy.deepcopy(cur_cont)

            cur_fea = {"geometry": {"coordinates": [coors]},
                       "properties": {"annotation_owner": "pred", "label_name": label_name}}

            features.append(cur_fea)

        to_save['features'] = features

        to_save = convert_int32_to_int(to_save)

        with open(os.path.join(save_dir, 'my_json', save_name + '.json'), 'w') as file:
            json.dump(to_save, file)
