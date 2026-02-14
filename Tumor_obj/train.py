import copy
import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn
from config import CFG
from tensorboardX import SummaryWriter
from torch.cuda import amp
from tqdm import tqdm

from utils import dice_coef, iou_coef, make_vis, muti_stage_loss_fusion
from dataset import prepare_dataloader

writer = SummaryWriter()
train_global_step = 0
train_global_step_4grid_save = 1  # 由于服务器无法直接使用tb，所以改为每隔固定个step就保存一次预测可视化结果
valid_global_step = 0
valid_global_step_4grid_save = 1

if CFG.resume and CFG.pretrained_weights:
    best_dice = float(CFG.pretrained_weights.split(CFG.sep)[-1].split('-')[-2].split('_')[0])
    best_jaccard = float(CFG.pretrained_weights.split(CFG.sep)[-1].split('-')[-1].split('_')[0])
    print('Weights loaded,current best dice=', best_dice)
else:
    best_dice = -np.inf
    best_jaccard = -np.inf

# 占位
best_epoch = -1
best_iter = -1
best_val_loss = np.inf


def train_one_epoch(model, criterion, optimizer, scheduler, dataloader, valid_loader, device, epoch):
    global train_global_step
    global train_global_step_4grid_save

    global best_dice, best_jaccard, best_epoch, best_iter, best_val_loss

    model = model.to(device, dtype=torch.float)
    model.train()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, sample in pbar:
        images, masks = sample['image'], sample['label']
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        if CFG.model_type == 'u2net' or CFG.model_type == 'u2net_p':
            y_pred, d1, d2, d3, d4, d5, d6 = model(images)
            loss2, loss = muti_stage_loss_fusion(y_pred, d1, d2, d3, d4, d5, d6, masks)
        else:
            y_pred = model(images)
            loss = criterion(y_pred, masks)

        if CFG.record_log:
            writer.add_scalar('Loss/iter-train', loss.item(), train_global_step)
            grid = make_vis('train', train_global_step, images, masks, y_pred)
            writer.add_image('train: gt vs pred', grid, train_global_step)
            train_global_step = train_global_step + 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        #################################################################################################################

        # 训练过程中产生的预测可视化结果保存
        if train_global_step_4grid_save % CFG.vis_freq == 0:
            make_vis('train', train_global_step_4grid_save, images, masks, y_pred, True)

        # 做验证, 并对当前epoch的当前iter的模型权重进行保存(条件：指标升高)
        if train_global_step_4grid_save % CFG.valid_freq == 0:
            model.eval()  ######切换到val状态
            val_loss, val_scores = valid_one_epoch(train_global_step_4grid_save, model, criterion, optimizer,
                                                   valid_loader, device=CFG.device, epoch=epoch)

            val_scores = val_scores.tolist()
            val_dice, val_jaccard = val_scores
            print('val_dice: ', val_dice)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print('\nfind lower val loss={},saving model\n'.format(val_loss))
                torch.save(model.state_dict(), os.path.join(CFG.weight_path,
                                                            CFG.model_type + "_val_loss-{}_val dice-{}_val jaccard-{}_best_epoch-{}_best_iter-{}.bin".format(
                                                                val_loss, val_dice, val_jaccard, best_epoch,
                                                                best_iter)))

            with open(CFG.log_dir, 'a+') as log_file:
                log_file.write(
                    'epoch = {}  train_iter = {}  train_epoch_loss = {}  val_loss = {}  val_dice = {}  val_jaccard = {}  lr = {}\n'.format(
                        epoch, train_global_step_4grid_save, epoch_loss, val_loss, val_dice, val_jaccard,
                        optimizer.param_groups[0]['lr']))

            if val_dice > best_dice:
                print(f"Valid Dice Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
                best_dice = val_dice
                best_jaccard = val_jaccard
                best_epoch = epoch
                best_iter = train_global_step_4grid_save

                PATH = os.path.join(CFG.weight_path,
                                    CFG.model_type + "_best_epoch-{}_best_iter-{}_val_loss-{}_best_dice-{}_best_jaccard-{}_zyj.bin".format(
                                        best_epoch, best_iter, val_loss, best_dice, best_jaccard))
                if best_dice > 0.6:
                    torch.save(model.state_dict(), PATH)
                print(
                    "better weights : best_epoch-{}_best_iter-{}_val_loss-{}_best_dice-{}_best_jaccard-{}_zyj.bin".format(
                        best_epoch, best_iter, val_loss, best_dice, best_jaccard))
                print();
                print()

            model.train()  ######切换回train状态

            if scheduler is not None:
                scheduler.step()

        train_global_step_4grid_save = train_global_step_4grid_save + 1
        #################################################################################################################

        #         if scheduler is not None:
        #             scheduler.step()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(train_global_step_4grid_save, model, criterion, optimizer, dataloader, device, epoch):
    global valid_global_step
    global valid_global_step_4grid_save

    model.eval()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, sample in pbar:
        images, masks = sample['image'], sample['label']
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        if CFG.model_type == 'u2net' or CFG.model_type == 'u2net_p':
            y_pred, d1, d2, d3, d4, d5, d6 = model(images)
            loss2, loss = muti_stage_loss_fusion(y_pred, d1, d2, d3, d4, d5, d6, masks)
        else:
            y_pred = model(images)
            loss = criterion(y_pred, masks)

        if CFG.record_log:
            writer.add_scalar('Loss/iter-valid', loss.item(), valid_global_step)
            grid = make_vis('valid', valid_global_step, images, masks, y_pred)
            writer.add_image('valid: gt vs pred', grid, valid_global_step)
            valid_global_step = valid_global_step + 1

        #         if valid_global_step_4grid_save%CFG.vis_freq==0:
        #             make_vis('valid',train_global_step_4grid_save,images,masks,y_pred,True)
        valid_global_step_4grid_save = valid_global_step_4grid_save + 1
        make_vis('valid', train_global_step_4grid_save, images, masks, y_pred, True)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')
    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores


def run_train(model, loss_fn, optimizer, scheduler, num_epochs, train_img_path, train_mask_path, valid_img_path,
              valid_mask_path, debug):
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()

    for epoch in range(1, num_epochs + 1):
        gc.collect()

        train_loader, valid_loader = prepare_dataloader(epoch, train_img_path, train_mask_path, valid_img_path,
                                                        valid_mask_path, debug)

        print(f'Epoch {epoch}/{num_epochs}', end='\n')

        train_loss = train_one_epoch(model, loss_fn, optimizer, scheduler, train_loader, valid_loader,
                                     device=CFG.device, epoch=epoch)

        # if scheduler is not None:
        #     scheduler.step(best_dice)
        if epoch % CFG.train_weight_save_freq == 0:
            print('epoch={}, saving ckpt...    saved!'.format(epoch))
            PATH = os.path.join(CFG.weight_path,
                                '{}_epoch-{}_trainLoss-{}.bin'.format(CFG.model_type, epoch, train_loss))
            torch.save(model.state_dict(), PATH)
        print();
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
