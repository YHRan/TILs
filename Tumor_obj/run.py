import torch
from config import CFG
from dataset import prepare_dataloader
from net import net
from train import run_train
from utils import criterion, fetch_scheduler


def main(net, train_img_path, train_mask_path, valid_img_path, valid_mask_path, debug):
    optimizer = torch.optim.Adam(net.parameters(), CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)
    loss_fn = criterion

    run_train(net, loss_fn, optimizer, scheduler, CFG.epochs, train_img_path, train_mask_path, valid_img_path,
              valid_mask_path, debug)


if __name__ == '__main__':
    train_mask_path = CFG.train_mask_path
    train_img_path = CFG.train_img_path

    valid_mask_path = CFG.valid_mask_path
    valid_img_path = CFG.valid_img_path

    if CFG.pretrained_weights:
        net.load_state_dict(torch.load(CFG.pretrained_weights), strict=True)
        print('pretrained weights loaded from {}'.format(CFG.pretrained_weights))

    print('--------------------------------config--------------------------------')
    print('train_img_size: ', CFG.train_img_size_)
    print('valid_img_size: ', CFG.valid_img_size)
    print('model_type: ', CFG.model_type)
    print('train_bs: ', CFG.train_bs)
    print('scheduler: ', CFG.scheduler)
    print('Ir: ', CFG.lr)
    print('----------------------------------------------------------------------')
    main(net, train_img_path, train_mask_path, valid_img_path, valid_mask_path, CFG.debug)
