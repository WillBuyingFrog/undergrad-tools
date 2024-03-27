import cv2
import os
import argparse
import numpy as np
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim


def calculate_entropy(img, mode=1, img0=None):
    if mode == 1:
        # 计算图像的信息熵
        entropy = shannon_entropy(img)
        return entropy
    elif mode == 2:
        assert img0 is not None
        # 计算图像的结构相似性
        ssim_value = ssim(img0, img, multichannel=True, channel_axis=2)
        return ssim_value



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Entropy demo')
    parser.add_argument('--origin_dir', default='roi_images/', type=str)
    parser.add_argument('--blur_dir', default='roi_images/optimize_fovea/', type=str)
    parser.add_argument('--test_compression', default=0, type=int, help='若为1，则忽略blur_dir，而是计算origin_dir中图片压缩再放大后的信息熵')
    parser.add_argument('--compress_ratio', default=2.0, type=float)
    parser.add_argument('--metric_mode', default=1, type=int)

    args = parser.parse_args()

    origin_dir = args.origin_dir
    blur_dir = args.blur_dir
    test_compression = args.test_compression
    compress_ratio = args.compress_ratio
    metric_mode = args.metric_mode
    
    if test_compression == 1:

        # 遍历origin_dir目录下的所有jpg图片，不包含origin_dir下任何层次的子文件夹中的图片
        for file in os.listdir(origin_dir):
            if file.endswith('.jpg'):
                origin_image_path = os.path.join(origin_dir, file)
                img0 = cv2.imread(origin_image_path)
                img1 = cv2.resize(img0, (int(img0.shape[1] / compress_ratio), int(img0.shape[0] / compress_ratio)))
                # 再把img1放大回img0的大小，从而实现图像压缩
                img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))
                if metric_mode == 1:
                    origin_entropy = calculate_entropy(img0)
                    compressed_entropy = calculate_entropy(img1)
                    print(f'[{file}] origin entropy: {origin_entropy}, compressed entropy: {compressed_entropy}')
                elif metric_mode == 2:
                    ssim_value = calculate_entropy(img1, mode=2, img0=img0)
                    print(f'[{file}] SSIM value: {ssim_value}')
                    
                # 保存img1到blur_dir中
                cv2.imwrite(os.path.join(blur_dir, file), img1)

    elif test_compression == 2:
         for file in os.listdir(origin_dir):
            if file.endswith('.jpg'):
                origin_image_path = os.path.join(origin_dir, file)

                gaussian_arg = int(compress_ratio)

                img0 = cv2.imread(origin_image_path)
                img1 = cv2.GaussianBlur(img0, (gaussian_arg, gaussian_arg), 0)
                if metric_mode == 1:
                    origin_entropy = calculate_entropy(img0)
                    compressed_entropy = calculate_entropy(img1)
                    print(f'[{file}] origin entropy: {origin_entropy}, compressed entropy: {compressed_entropy}')
                elif metric_mode == 2:
                    ssim_value = calculate_entropy(img1, mode=2, img0=img0)
                    print(f'[{file}] SSIM value: {ssim_value}')
                
                
                # 保存img1到blur_dir中
                cv2.imwrite(os.path.join(blur_dir, file), img1)
    elif test_compression == 0:
        origin_image_paths = []
        blur_image_paths = []
        for file in os.listdir(origin_dir):
            if file.endswith('.jpg'):
                origin_image_path = os.path.join(origin_dir, file)
                origin_image_paths.append(origin_image_path)
            
        for file in os.listdir(blur_dir):
            if file.endswith('.jpg'):
                blur_image_path = os.path.join(blur_dir, file)
                blur_image_paths.append(blur_image_path)

        print(f'Total {len(origin_image_paths)} origin images')
        print(f'Total {len(blur_image_paths)} blur images')
        
        assert len(origin_image_paths) == len(blur_image_paths)

        for i in range(len(origin_image_paths)):
            img0 = cv2.imread(origin_image_paths[i])
            img1 = cv2.imread(blur_image_paths[i])
            if metric_mode == 1:
                origin_entropy = calculate_entropy(img0)
                compressed_entropy = calculate_entropy(img1)
                print(f'[{origin_image_paths[i]}] origin entropy: {origin_entropy}, compressed entropy: {compressed_entropy}')
            elif metric_mode == 2:
                ssim_value = calculate_entropy(img1, mode=2, img0=img0)
                print(f'[{origin_image_paths[i]}] SSIM value: {ssim_value}')