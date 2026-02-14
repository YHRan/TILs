import glob
import shutil

import cv2
import numpy as np
import spams


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

if __name__ =='__main__':
    # 单张图增强（离线）
    # img_path=r'E:\ProjectSpaces\seg\xueguanData\patchData\img-bowl\ST19Rm-TE-226-1-000001_(7120, 10694).png'
    # img = cv2.imread(img_path)
    # scale_factor=0.4
    # img_sn = stain_separate(img, scale_factor, False)
    # cv2.imwrite(r'E:\ProjectSpaces\seg\xueguanData\augPatchData\a.png', img_sn[:, :, ::-1])

    # 批量增强（离线) 只对训练集进行增强，验证集不动
    imgs_dir='E:\\ProjectSpaces\\seg\\xueguanData\\patchData\\trainSet\\img-bowl\\'
    masks_dir='E:\\ProjectSpaces\\seg\\xueguanData\\patchData\\trainSet\\0\\'
    all_img_paths=glob.glob(imgs_dir+'*')
    scale_list=[0.5,0.6,0.7, 0.8, 0.9, 1.1, 1.2,1.3,1.4,1.5]
    for scale in scale_list:
        for img_path in all_img_paths:
            base_name=img_path.split('\\')[-1]
            print(scale, base_name)
            aug_name='aug'+str(int(scale*10))+base_name
            # print(aug_name)
            img = cv2.imread(img_path)
            img_sn = stain_separate(img, scale, False)
            cv2.imwrite('E:\\ProjectSpaces\seg\\xueguanData\\augPatchData\\imgs\\'+aug_name, img_sn[:, :, ::-1])# BGR->RGB

            # 复制一份mask
            mask_path=masks_dir+base_name
            dst_path='E:\\ProjectSpaces\seg\\xueguanData\\augPatchData\\msks\\'+aug_name
            shutil.copyfile(mask_path,dst_path)