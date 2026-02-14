import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from augumentation import stain_separate
from config import CFG
from skimage import color, io, transform
from torch.utils.data import DataLoader, Dataset
from utils import load_img, load_msk, plot_batch, standard_scale_and_totensor


class BuildDataset(Dataset):
    def __init__(self, imgs_path_list, masks_path_list=[], transforms=None):
        self.mask_paths = masks_path_list
        self.img_paths = imgs_path_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = load_img(self.img_paths[idx])

        if (0 == len(self.mask_paths)):
            label = np.zeros(image.shape)
        else:
            label = load_msk(self.mask_paths[idx])

        sample = {'image': image, 'label': label}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


# 使用一些patch作为验证集
def prepare_dataloader(epoch, train_img_path, train_mask_path, valid_img_path, valid_mask_path, debug=False):
    train_imgs_ = sorted(glob.glob(train_img_path + '/*'))
    train_masks_ = sorted(glob.glob(train_mask_path + '/*'))
    valid_imgs = sorted(glob.glob(valid_img_path + '/*'))
    valid_masks = sorted(glob.glob(valid_mask_path + '/*'))

    if debug:
        print('Debug Mode...')
        train_imgs = train_imgs_[:10]
        train_masks = train_masks_[:10]
        valid_imgs = train_imgs_[10:]
        valid_masks = train_masks_[10:]
    else:
        train_imgs = train_imgs_
        train_masks = train_masks_

    print('train: ', len(train_imgs), '\tvalid: ', len(valid_imgs))

    assert len(train_imgs) == len(train_masks) and len(valid_imgs) == len(valid_masks)
    assert [msk_path.split('/')[-1] == img_path.split('/')[-1] for (msk_path, img_path) in zip(train_masks, train_imgs)]
    assert [msk_path.split('/')[-1] == img_path.split('/')[-1] for (msk_path, img_path) in zip(valid_masks, valid_imgs)]

    if CFG.mult_reso:
        # 不同的epoch阶段使用不同的分辨率进行训练(从小到大)
        if epoch <= 10:
            train_img_size = CFG.train_img_size[0]
        elif epoch <= 20:
            train_img_size = CFG.train_img_size[1]
        elif epoch <= 30:
            train_img_size = CFG.train_img_size[2]
        elif epoch <= 40:
            train_img_size = CFG.train_img_size[3]
        elif epoch <= 50:
            train_img_size = CFG.train_img_size[4]
        else:
            train_img_size = CFG.train_img_size[5]
        valid_img_size = train_img_size
    else:
        train_img_size = CFG.train_img_size_
        valid_img_size = CFG.valid_img_size

    print('Preparing dataloader for epoch {} ,train img size = {}'.format(epoch, train_img_size))

    data_transforms = CFG.data_transforms(train_img_size, valid_img_size)

    train_dataset = BuildDataset(train_imgs, train_masks, data_transforms['train'])
    valid_dataset = BuildDataset(valid_imgs, valid_masks, data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True,
                              drop_last=True)

    return train_loader, valid_loader


if __name__ == '__main__':
    train_mask_path = CFG.train_mask_path
    train_img_path = CFG.train_img_path
    aug_train_img_path = CFG.aug_train_img_path
    aug_train_mask_path = CFG.aug_train_mask_path
    valid_mask_path = CFG.valid_mask_path
    valid_img_path = CFG.valid_img_path

    train_loader, valid_loader = prepare_dataloader(train_img_path, train_mask_path, aug_train_img_path,
                                                    aug_train_mask_path, valid_img_path, valid_mask_path, CFG.debug)
    for i, sample in enumerate(train_loader):
        img, mask = sample['image'], sample['label']
        # print(img.shape,np.unique(img), mask.shape,np.unique(mask))
        plot_batch(img, mask)
        if i > 100:
            break
