import glob
import sys
sys.path.append('/home/fhs/data-fhs/Project-VesselSeg/src_v4/StainTools-master')
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import staintools


def stain_normalize(source_path,target_path):
    source = cv2.imread(source_path, cv2.COLOR_BGR2RGB )
    target = cv2.imread(target_path, cv2.COLOR_BGR2RGB )
    print(source)

    # run reinhard normalization
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(np.array(target))
    reinhard_normalized = normalizer.transform(np.array(source))
    # print(reinhard_normalized)

    plt.figure(figsize=(16,9))
    plt.subplot(131)
    plt.title(f"Source")
    plt.axis('off')
    plt.imshow(source)
    plt.subplot(132)
    plt.title(f'reinhard_normalized')
    plt.axis('off')
    plt.imshow(reinhard_normalized)
    plt.subplot(133)
    plt.title(f'target')
    plt.axis('off')
    plt.imshow(target)
    plt.show()

if __name__ =='__main__':
    sour_path=r'E:\ProjectSpaces\seg\xueguanData\augPatchData\imgs'
    tar_path=r'E:\ProjectSpaces\seg\xueguanData\style_template'# 风格图

    source_paths=glob.glob(sour_path+'/*')
    target_paths=glob.glob(tar_path+'/*')
    # print(source_paths)

    for source_path in source_paths:
        target_path=np.random.choice(target_paths)
        print(target_path)
        stain_normalize(source_path,target_path)