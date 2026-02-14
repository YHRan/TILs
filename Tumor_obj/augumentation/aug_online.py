import collections
import copy
import glob
import math
import random
import shutil
import os
import staintools

# import staintools
import imgaug
import cv2
import spams
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage import color, io, transform
from torch.utils.data import DataLoader, Dataset

from .randstainna import RandStainNA


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        # with rgb color
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:  # channel=1
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:  # channel=3
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # HWC->CHW
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(np.ascontiguousarray(tmpImg)),
                'label': torch.from_numpy(np.ascontiguousarray(tmpLbl))}


seq = iaa.Sequential([
    iaa.Fliplr(p=0.5),
    iaa.Flipud(p=0.5),
    iaa.Sometimes(0.5,
                  iaa.Crop(percent=(0, 0.12))
                  ),
#     iaa.ChannelShuffle(0.5),
#     iaa.Sometimes(0.5,
#             iaa.Sharpen((0.0, 0.2))
#         ),
#     iaa.LinearContrast((0.6, 1.4)),
#     iaa.Sometimes(0.5,
    #     iaa.Affine(rotate=(-10, 10),scale=1.2),
    # ), 
    #     iaa.Sometimes(0.5,
    #         iaa.Multiply((0.90, 1.22)),
    #     ),
    #     iaa.Sometimes(0.5,
    #     iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #             rotate=(-45, 45),
    #             scale=(0.6, 1.5))
    #     ),

])


# 基于imgaug
class Resize():
    def __init__(self, s):
        self.aug = iaa.Sequential([iaa.Resize(size=[s, s], interpolation='nearest')])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, axis=0).astype(np.float32) / 255
        label = np.expand_dims(label, axis=0).astype(np.int32)  # segmentation 需要 int 型

        image, label = self.aug(images=image, segmentation_maps=label)

        image = np.squeeze(image, 0)
        label = np.squeeze(label, 0)

        return {'image': image, 'label': label}


# 基于imgaug    
class RandomAug():
    def __init__(self, p=1):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, axis=0).astype(np.float32)
        label = np.expand_dims(label, axis=0).astype(np.int32)

        image, label = seq(images=image, segmentation_maps=label)

        image = np.squeeze(image, 0)
        image = np.clip(image, 0, 1)

        label = np.squeeze(label, 0)

        return {'image': image, 'label': label}


class StainNormalization():
    def __init__(self, target_paths):
        self.target_paths = target_paths

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        target_path = np.random.choice(self.target_paths)
        target = cv2.imread(target_path, cv2.COLOR_BGR2RGB)

        normalizer = staintools.ReinhardColorNormalizer()
        normalizer.fit(np.array(target))

        image = image * 255
        image = image.astype('uint8')

        reinhard_normalized = normalizer.transform(np.array(image))

        return {'image': reinhard_normalized, 'label': label}


def stain_separate(img_array, trans_scale=3, debug=False):
    """
    染色分离
    :param img_array: RGB原图
    :param trans_scale: 拉伸尺度
    :param debug: 是否保存效果图
    :return:染色归一化后RGB图；染色归一化后单苏木素RGB图；染色归一化后单伊红RGB图
    """

    def intensity_norm(img):
        p = np.percentile(img, 95)
        img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
        return img

    W_templ = np.array([[0.53728666, 0.02053517],
                        [0.75435933, 0.87660042],
                        [0.3771804, 0.48078063]])

    patch_array = intensity_norm(img_array)

    # getV
    Is = patch_array.reshape((-1, 3)).T
    Is[Is == 0] = 1
    Vs = np.log(255 / Is)

    # getH
    Hs = spams.lasso(np.asfortranarray(Vs), np.asfortranarray(W_templ), mode=2, lambda1=0.03,
                     pos=True, verbose=False, numThreads=10).toarray()

    # get sn image
    Hs_norm = Hs * trans_scale
    Vs_norm = np.dot(W_templ, Hs_norm)
    Is_norm = 255 * np.exp(-1 * Vs_norm)
    img_sn = Is_norm.T.reshape(patch_array.shape).astype(np.uint8)

    return img_sn


class StainDegreeAug():
    def __init__(self, scale_list=[0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]):
        self.scale_list = scale_list

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        scale = np.random.choice(self.scale_list)
        p = random.random()
        if p < 0.5:  # 随机做染色增强
            image = stain_separate(image, scale, False)

        return {'image': image, 'label': label}


# 新的随机染色增强方式，来源：https://github.com/yiqings/RandStainNA
class RandomStainAug():
    def __init__(self):
        current_directory = os.path.abspath(os.path.dirname(__file__))
        self.randstainna = RandStainNA(
            yaml_file=os.path.join(current_directory, 'CRC_LAB_randomTrue_n0.yaml'),
            std_hyper=0.0,
            distribution='normal',
            probability=1.0,
            is_train=True
        )

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        p = random.random()
        if p < 0.5:  # 随机做染色增强
            image = self.randstainna(image)

        return {'image': image, 'label': label}


#########################################################################################################################
# 也可以离线制作增强数据(在线增强使用imgaug慢一些)
if __name__ == "__main__":
    show = True

    imgs_dir = r'D:\program\src\data\keliguanxibaohe\train\img'
    masks_dir = r'D:\program\src\data\keliguanxibaohe\train\mask'
    # imgs_dir = 'E:\\ProjectSpaces\\seg\\xueguanData\\patchData\\trainSet\\img-bowl\\'
    # masks_dir = 'E:\\ProjectSpaces\\seg\\xueguanData\\patchData\\trainSet\\0\\'
    all_img_paths = glob.glob(imgs_dir + '*')
    all_msk_paths = glob.glob(masks_dir + '*')

    times = 20  # 增强倍数

    if not show:
        from aug_local import stain_separate

        scale_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]

        for i in range(times):
            for img_path, msk_path in zip(all_img_paths, all_msk_paths):
                assert img_path.split('\\')[-1] == msk_path.split('\\')[-1]

                base_name = img_path.split('\\')[-1]
                print(i, base_name)
                # print(aug_name)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                p = random.random()
                if p < 0.5:  # 随机做染色增强
                    scale = np.random.choice(scale_list)
                    aug_name = 'augT' + str(int(i)) + 'Scale' + str(int(scale * 10)) + base_name
                    img = stain_separate(img, scale, False)
                else:
                    aug_name = 'augT' + str(int(i)) + 'Scale10' + base_name

                img = np.expand_dims(img, axis=0).astype(np.float32) / 255

                msk = cv2.imread(msk_path)
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
                msk = np.expand_dims(np.expand_dims(msk, axis=0), axis=-1).astype(np.int32)
                # print(msk.shape)

                img_aug, msk_aug = seq(images=img, segmentation_maps=msk)

                img_aug = np.squeeze(img_aug, 0)
                img_aug = np.clip(img_aug, 0, 255)
                msk_aug = np.squeeze(msk_aug, 0)
                # print(msk_aug.shape)

                # cv2.imwrite('E:\\ProjectSpaces\seg\\xueguanData\\augPatchData\\imgs\\' + aug_name, img_aug * 255)
                # cv2.imwrite('E:\\ProjectSpaces\seg\\xueguanData\\augPatchData\\msks\\' + aug_name, msk_aug * 255)
                cv2.imwrite(r'D:\program\src\data\pseudo_patch\train\img' + aug_name, img_aug * 255)
                cv2.imwrite(r'D:\program\src\data\pseudo_patch\train\mask' + aug_name, msk_aug * 255)
    else:
        # 单张图片增强结果可视化
        for i in range(100):
            img_path = np.random.choice(all_img_paths)
            img = cv2.imread(img_path)
            ori_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = copy.deepcopy(ori_img)

            img = np.expand_dims(img, axis=0).astype(np.float32) / 255
            # print(ori_img)

            msk_path = os.path.join('\\'.join(imgs_dir.split('\\')[:-2]), '0', img_path.split('\\')[-1])
            print(msk_path)

            ori_msk = cv2.imread(msk_path)
            msk = cv2.cvtColor(ori_msk, cv2.COLOR_BGR2GRAY)
            msk = np.expand_dims(np.expand_dims(msk, axis=0), axis=-1).astype(np.int32)  # segmentation 需要 int 型

            img_aug, msk_aug = seq(images=img, segmentation_maps=msk)
            print(img_aug)

            img_aug = np.squeeze(img_aug, 0)
            img_aug = np.clip(img_aug, 0, 255)
            msk_aug = np.squeeze(msk_aug, 0)
            # cv2.imwrite(r'E:\ProjectSpaces\seg\src_tensorboard\a.png',img_aug*255)# 保存时要*255
            # cv2.imwrite(r'E:\ProjectSpaces\seg\src_tensorboard\a.png',msk_aug*255)# 保存时要*255

            plt.subplot(2, 2, 1)
            plt.imshow(ori_img)
            plt.subplot(2, 2, 2)
            plt.imshow(ori_msk)
            plt.subplot(2, 2, 3)
            plt.imshow(img_aug)
            plt.subplot(2, 2, 4)
            plt.imshow(msk_aug)
            plt.show()
