import openslide
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_patch_coordinates(width, height, patch_size=(640, 640), stride=(640, 640)):
    """
    1. 列出算术框架
    2. 从内到外组装、重构
    """

    coords = []

    y1 = 0
    while y1 <= height - patch_size[1]:
        y2 = y1 + patch_size[1]

        x1 = 0
        while x1 <= width - patch_size[0]:
            x2 = x1 + patch_size[0]
            coords.append([(x1, y1), (x2, y2)])
            x1 = x1 + stride[0]
        y1 = y1 + stride[1]

    return coords


def should_save_patch(patch_array, threshold=230):
    """判断是否应该保存这个块."""
    gray_patch = Image.fromarray(patch_array).convert('L')
    mean_val = np.mean(np.array(gray_patch))
    return mean_val < threshold


def downsample_image_to_factor(slide, target_downsample, width, height):
    """手动降采样图像."""
    target_w, target_h = int(width / target_downsample), int(height / target_downsample)
    img = slide.read_region((0, 0), 0, (width, height)).convert("RGBA")
    img_downsampled = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img_downsampled)[:, :, :3]  # Convert PIL image to numpy array and drop alpha channel


def extract_patches_from_svs(svs_path, save_dir, target_downsample, patch_size, stride, roi_region=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    svs_name = os.path.basename(svs_path).split('.')[0]
    print("The current filename is -->", svs_name)
    slide = openslide.OpenSlide(svs_path)
    level_count = slide.level_count
    for level in range(level_count):
        width, height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        print(f'Level {level}: {width}x{height}, downsample factor: {downsample}')

    width, height = slide.level_dimensions[0]  # Always use level 0 for manual downsampling

    # Manual downsampling if needed
    if target_downsample != 1:
        whole_slide_img = downsample_image_to_factor(slide, target_downsample, width, height)
    else:
        whole_slide_img = np.array(slide.read_region((0, 0), 0, (width, height)).convert("RGB"))

    # Adjust coordinates if using ROI
    if roi_region:
        (x1, y1), (x2, y2) = roi_region
        whole_slide_img = whole_slide_img[y1:y2, x1:x2]

    coords = get_patch_coordinates(*whole_slide_img.shape[:2], patch_size, stride)

    for i, ((x1, y1), (x2, y2)) in tqdm(enumerate(coords)):
        patch_array = whole_slide_img[y1:y2, x1:x2]

        if should_save_patch(patch_array):
            patch = Image.fromarray(patch_array)
            patch_path = os.path.join(save_dir, f"{svs_name}_patch_{i}.png")
            patch.save(patch_path)


if __name__ == '__main__':
    # Example usage:
    svs_path = "/home/yyj/Inference/MD-single-process/ReviewData/svs/ST18Rf-LG-MD-ML-PG-SD-316-1-000016.svs"
    save_dir = "/home/yyj/General/temp"
    target_downsample = 2  # Specify the target downsample factor
    patch_size = (1280, 1280)
    stride = (960, 960)
    # 以下二选一
    x1, y1 = 48000, 10000
    x2, y2 = 63500, 35800
    roi_region = ((x1, y1), (x2, y2))  # Define ROI or set to None

    extract_patches_from_svs(svs_path, save_dir, target_downsample, patch_size, stride, roi_region)
