import glob
import os
import cv2
import torch
from augumentation import RandomAug, Resize, ToTensorLab, RandomStainAug
from torchvision import transforms


class CFG:
    ##########################################################################################################
    # project_dir = '/home/fhs/data-fhs/2024S1_SDEV/fresh_version'
    project_dir = '/home/ws/data-ws/big_mouse/TH/blood'

    exp_name = 'ep_wlk'
    model_type = 'u2net_p'

    # 训练数据路径
    train_mask_path = os.path.join(project_dir, 'ep_wlk/train_mask')
    train_img_path = os.path.join(project_dir, 'ep_wlk/train')

    # 验证数据路径
    valid_mask_path = os.path.join(project_dir, 'ep_wlk/val_mask')
    valid_img_path = os.path.join(project_dir, 'ep_wlk/val')

    train_bs = 4
    valid_bs = 4
    lr = 6e-3
    T = 1
    epochs = 300 * T  #200*
    ##########################################################################################################

    PATCH_SIZE = 300
    mult_reso = False  # 是否开启多分辨率训练
    train_img_size_ = 1280  # 固定分辨率训练的尺寸,1024
    train_img_size = [256, 288, 320, 384, 480, 512]  # 多分辨率训练的尺寸列表
    valid_img_size = 1280 # 与训练保持一致
    train_weight_save_freq = 1000

    sep = '/'
    record_log = False
    debug = False
    resume = False

    # 训练权重保存路径
    weight_path = os.path.join(project_dir, 'exp', exp_name, 'save_models')
    # 训练过程中间可视化结果输出路径
    vis_dir = os.path.join(project_dir, 'exp', exp_name, 'vis_dir')
    log_dir_ = os.path.join(project_dir, 'exp', exp_name, 'log_dir')
    # 自动创建上述文件夹
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(log_dir_, exist_ok=True)
    log_dir = os.path.join(log_dir_, 'log.txt')

    if debug:
        valid_freq = 125
        vis_freq = 125
    else:
        num_train_imgs = len(os.listdir(train_mask_path))
        t = num_train_imgs // train_bs
        valid_freq = t * T  # 每隔valid_freq个iter做一次验证
        vis_freq = t * T  # 每隔vis_freq个iter保存一次预测可视化结果对比图(训练+验证都有设置)

    wd = 0
    min_lr = 9e-4
    T_max = 10  # 在使用余弦退火调度器时，以2*T_max个iter（或者epoch，看调度器放哪儿）为一个周期，即第0到T_max个iter过程中，学习率由lr下降至min_lr，在第T_max到第2*T_max个uiter过程中，学习率由min_lr恢复到lr
    T_0 = 1000  # 使用带热重启的余弦退火调度器时，以T_0作为初始的热重启周期(iter或者epoch)，后续T_0=T_0 * T_mult
    T_mult = 2
    scheduler = 'ExponentialLR'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_accumulate = 1  # 1

    # 风格图路径（已弃用）
    # style_dir=glob.glob('/home/fhs/data-fhs/Project-VesselSeg/xueguanDataNew/style_template'+'/*')

    # 染色增强配置文件路径
    stain_cfg = './augumentation/CRC_LAB_randomTrue_n0.yaml'

    # 配置预训练权重
    if model_type == 'u2net_p':
        pretrained_weights = './pretrained_weights/u2netp.pth'
#         pretrained_weights = '/home/ws/data-ws/big_mouse/TH/blood/exp/th_cell2/save_models/u2net_best.bin'
    elif model_type == 'u2net':
        pretrained_weights = './pretrained_weights/u2net.pth'
#         pretrained_weights = '/home/ws/data-ws/big_mouse/TH/blood/exp/th_cell2/save_models/u2net_best.bin'
    else:
        pretrained_weights = None

    # 配置数据增强
    @staticmethod
    def data_transforms(train_img_size, valid_img_size):
        t = {
            'train':
                transforms.Compose([
#                     StainDegreeAug(),# 随机染色增强，比较耗时
                    RandomStainAug(),  # 随机染色增强的另一种实现，速度较快
                    Resize(train_img_size),
                    RandomAug(),  # 离线做也可
                    # StainNormalization(style_dir),（已弃用）
                    ToTensorLab()
                ]),
            'valid':
                transforms.Compose([
                    Resize(valid_img_size),
                    # StainNormalization(style_dir),（已弃用）
                    ToTensorLab()
                ])
        }
        return t


if __name__ == '__main__':
    print(CFG.data_transforms(512, 512))
