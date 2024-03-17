import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

def precompute_distance(shape):
    center = (shape[1] / 2, shape[0] / 2)
    # Compute the distance from the center for each pixel
    y = np.arange(shape[0]).reshape(-1, 1)
    x = np.arange(shape[1])
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)
    
    return distances, max_radius

def blur_based_on_distance(img_rgb, alpha):
    # Downsample the image by 16x and then upsample
    img_downsampled = cv2.resize(img_rgb, (img_rgb.shape[1] // 16, img_rgb.shape[0] // 16))
    img_blur = cv2.resize(img_downsampled, (img_rgb.shape[1], img_rgb.shape[0]))
    img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)[:,:,None]

    # img_adjusted = (1-alpha) * img_blur + alpha * img_rgb
    img_adjusted = img_blur_gray + alpha * (img_rgb - img_blur_gray*1.0)
    img_adjusted = img_adjusted.astype(np.uint8)
    
    return img_adjusted

if __name__ == "__main__":
    image_path = r"./img_test.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR图像转换为RGB格式，以便正确显示

    # Precompute the distances and levels only once 避免在处理每个图像时都重新计算距离和级别矩阵，从而提高效率。
    distances, max_radius = precompute_distance(shape=image.shape)
    blur_decay = np.exp(-distances / max_radius * 2)[:,:,None]
    # blur_decay = (1-distances / max_radius)[:,:,None] * .1

    # Measure the time before the function call
    start_time = time.time()
    result = blur_based_on_distance(image, blur_decay)

    # Measure the time after the function call
    end_time = time.time()
    # Compute the elapsed time
    elapsed_time = end_time - start_time

    print(elapsed_time)
    # Create a figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Show the original image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('on')

    # Show the processed image
    ax[1].imshow(result, cmap='gray')
    ax[1].set_title('Processed Image')
    ax[1].axis('on')


    plt.tight_layout()
    plt.show()
