import copy
import os
from collections import Counter
# from symbol import global_stmt
import sys

sys.path.append('./region-mutual-information-pytorch-main')
import torch
# from rmi import RMILoss

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
from config import CFG
from PIL import Image
from torch.optim import lr_scheduler

JaccardLoss = smp.losses.JaccardLoss(mode='binary')
DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode='binary', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)
# rmi_loss = RMILoss(with_logits=True)# region-mutual-information loss，基于互信息

# for u2net
bce_loss = nn.BCELoss(size_average=True)  # with_logits=False 默认

def muti_stage_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
# 	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item()))

	return loss0, loss

# bce_loss=smp.losses.SoftBCEWithLogitsLoss()
# def muti_stage_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
#     loss0 = bce_loss(d0, labels_v) + DiceLoss(d0, labels_v)
#     loss1 = bce_loss(d1, labels_v) + DiceLoss(d1, labels_v)
#     loss2 = bce_loss(d2, labels_v) + DiceLoss(d2, labels_v)
#     loss3 = bce_loss(d3, labels_v) + DiceLoss(d3, labels_v)
#     loss4 = bce_loss(d4, labels_v) + DiceLoss(d4, labels_v)
#     loss5 = bce_loss(d5, labels_v) + DiceLoss(d5, labels_v)
#     loss6 = bce_loss(d6, labels_v) + DiceLoss(d6, labels_v)

#     loss = 0.5 * (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6)
#     # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

#     return loss0, loss


def criterion(y_pred, y_true):
    #     y_pred=torch.sigmoid(y_pred)
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)
    # return nn.CrossEntropyLoss(reduction='sum')


#     return 0.5*rmi_loss(y_pred, y_true)+0.5*BCELoss(y_pred, y_true)

def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        print('using CosineAnnealingLR')
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=CFG.T_mult,
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        print('using ReduceLROnPlateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=6, threshold=0.0001,
                                                   min_lr=CFG.min_lr)
    elif CFG.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.996) # gamma=0.988
    elif not CFG.scheduler:
        return None

    return scheduler


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    if CFG.model_type == 'smp':
        y_pred = torch.sigmoid(y_pred)  # 添加一个映射到01之间
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    if CFG.model_type == 'smp':
        y_pred = torch.sigmoid(y_pred)  # 添加一个映射到01之间
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def standard_scale_and_totensor(img):
    img = img / 255
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    return img


# 适用于二分类且mask是[h,w,3]的，[h,w,0]==[h,w,1]==[h,w,2].
def load_msk(path):
    msk = np.array(Image.open(path))
    if msk.shape[-1] == 3:
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    msk = msk[:, :, np.newaxis]
    # print('msk',msk.shape)# (512, 512, 1)
    if 255 in set(np.unique(msk)):
        msk = msk // 255
    return msk  # 因为类别是只有1类，用255表示，所以可以直接除以255，当类别数大于1时，可以考虑直接映射到[1,2,3...]


def show_img(img, mask=None):
    plt.imshow(img, cmap='bone')

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
    # plt.axis('off')


def plot_batch(imgs, msks, size=4):
    plt.figure(figsize=(5 * 4, 5))
    for idx in range(size):
        plt.subplot(1, 4, idx + 1)

        img = imgs[idx,].permute((1, 2, 0)).numpy() * 255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy() * 255.0
        msk = msk.astype('uint8')

        show_img(img, msk)
    plt.tight_layout()
    plt.show()


def make_vis(indicator, global_step, images, masks, y_pred, save_grid=False):
    size = CFG.train_bs

    imgs = copy.deepcopy(images)

    for idx in range(size):
        img = imgs[idx,].permute(1, 2, 0).cpu().detach().numpy()
        img[:, :, 0] = img[:, :, 0] * 0.229 + 0.485
        img[:, :, 1] = img[:, :, 1] * 0.224 + 0.456
        img[:, :, 2] = img[:, :, 2] * 0.225 + 0.406
        img = img * 255
        img = img.astype('uint8')

        msk = masks[idx,].permute((1, 2, 0)).cpu().detach().numpy() * 255.0
        msk = msk.astype('uint8')

        y_pred_ = y_pred[idx,].permute((1, 2, 0)).cpu().detach()

        if CFG.model_type in ['mobile_sam', 'smp']:
            y_pred_ = torch.sigmoid(y_pred_)

        pred_msk = y_pred_.numpy()

        pred_msk[pred_msk >= 0.5] = 255
        pred_msk[pred_msk < 0.5] = 0

        pred_msk = pred_msk.astype('uint8')

        gt = np.concatenate((msk, msk, msk), axis=-1)
        pred = np.concatenate((pred_msk, pred_msk, pred_msk), axis=-1)

        grid = np.hstack((gt, img, pred)).transpose((2, 0, 1))

        # grid=grid[::-1,:,:]

        if save_grid:
            vis_dir = os.path.join(CFG.vis_dir, indicator)
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            save_name = str(global_step) + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(vis_dir, save_name), grid.transpose((1, 2, 0)))

    return grid
