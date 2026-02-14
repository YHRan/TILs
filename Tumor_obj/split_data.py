import os
import argparse
import random
import shutil
from shutil import copyfile
import glob
from PIL import Image


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def main(config):
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    # rm_mkdir(config.test_path)
    # rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data_list = []
    GT_list = []
    # for filename in filenames:
    #     ext = os.path.splitext(filename)[-1]
    #     if ext == '.png':
    #         filename = filename.split('train')[-1]
    #         data_list.append('train/' + filename)
    #         GT_list.append('label/' + filename)
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '.png':
            # filename = filename.split('train')[-1]
            data_list.append(filename)
            GT_list.append(filename)

    num_total = len(data_list)
    num_train = int(config.train_ratio * num_total)
#     num_valid = int(config.valid_ratio * num_total)
    num_valid = num_total - num_train
    # num_test = num_total - num_train - num_valid

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)
    # print('\nNum of test set : ', num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()
        print(idx)
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_path, data_list[idx])
        copyfile(src, dst)
        print(src, dst)
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_GT_path, GT_list[idx])
        copyfile(src, dst)
        print(src, dst)
        printProgressBar(i + 1, num_train, prefix='Producing train set:', suffix='Complete', length=50)

    for i in range(num_valid):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.valid_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.valid_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_valid, prefix='Producing valid set:', suffix='Complete', length=50)

    # for i in range(num_test):
    #     idx = Arange.pop()
    #
    #     src = os.path.join(config.origin_data_path, data_list[idx])
    #     dst = os.path.join(config.test_path, data_list[idx])
    #     copyfile(src, dst)
    #
    #     src = os.path.join(config.origin_GT_path, GT_list[idx])
    #     dst = os.path.join(config.test_GT_path, GT_list[idx])
    #     copyfile(src, dst)
    #
    #     printProgressBar(i + 1, num_test, prefix='Producing test set:', suffix='Complete', length=50




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.9)
#     parser.add_argument('--valid_ratio', type=float, default=0.02)
    # parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default=r'/home/zyj/little-mouse/SD/xianpao/data/cover/img')
    parser.add_argument('--origin_GT_path', type=str, default=r'/home/zyj/little-mouse/SD/xianpao/data/cover/mask')

    parser.add_argument('--train_path', type=str, default='/home/zyj/little-mouse/SD/xianpao/data/ds_json/train/')
    parser.add_argument('--train_GT_path', type=str, default='/home/zyj/little-mouse/SD/xianpao/data/ds_json/train_mask/')
    parser.add_argument('--valid_path', type=str, default='/home/zyj/little-mouse/SD/xianpao/data/ds_json/val/')
    parser.add_argument('--valid_GT_path', type=str, default='/home/zyj/little-mouse/SD/xianpao/data/ds_json/val_mask/')
    # parser.add_argument('--test_path', type=str, default='./dataset/test/')
    # parser.add_argument('--test_GT_path', type=str, default='./dataset/test_labels/')

    config = parser.parse_args()
    print(config)
    main(config)
