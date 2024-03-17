import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

def decrease_saturation_based_on_distance(result, distances, max_radius):
    # Convert the result back to 8-bit integers
    result_rgb = np.clip(result, 0, 255).astype(np.uint8)

    # Convert the result to RGB color space, then to HSV color space
    result_hsv = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2HSV).astype(float)

    # Compute the saturation scaling factor based on the distance from the center
    exp_decay = np.exp(-distances / max_radius * 1)

    # Decrease color saturation based on the distance from the center
    result_hsv[:, :, 1] *= exp_decay

    # Convert the result back to RGB color space
    result_rgb = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return result_rgb

def precompute_distance_and_levels(shape, num_levels):
    center = (shape[1] / 2, shape[0] / 2)
    # Compute the distance from the center for each pixel
    y = np.arange(shape[0]).reshape(-1, 1)
    x = np.arange(shape[1])
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)
    
    # Convert the distance to a level in the range [0, num_levels)
    levels = np.clip((np.power(distances / max_radius, 1.5) * num_levels).astype(np.int32), 0, num_levels - 1)  # equal_ratio
    
    return distances, levels

def pyrDown(image, times=1):
    """Applies cv2.pyrDown operation on the input image a number of times."""
    for _ in range(times):
        image = cv2.pyrDown(image)
    return image

def pyrUp(image, times=1, dstsize=None):
    """Applies cv2.pyrUp operation on the input image a number of times."""
    for i in range(times):
        if i == times - 1:  # If this is the last iteration
            image = cv2.pyrUp(image, dstsize=dstsize)
        else:
            image = cv2.pyrUp(image)
    return image

# create a sequence of blurred images by downsampling and upsampling in RGB space
def create_pyramid(image, num_levels):
    blurred = [image.astype(np.float64)]
    for i in range(1, num_levels):
        # Downsample the image
        down = pyrDown(blurred[-1])
        # Upsample the image and resize to original image size
        up = pyrUp(down, dstsize=(blurred[0].shape[1], blurred[0].shape[0]))
        blurred.append(up)
    return blurred

def simulate_fovea(image: Image.Image, distances, levels, blurness_level: int = 5) -> Image.Image:
    # Convert the image to a numpy array
    image_array = np.array(image)
    blurred = create_pyramid(image_array, blurness_level)

    # Initialize result with the same shape as the input image
    result = np.zeros_like(image_array, dtype=np.float64)

    # Process the image in blocks to save memory
    block_size = 200  # Use larger blocks
    for i in range(0, image_array.shape[0], block_size):
        for j in range(0, image_array.shape[1], block_size):
            block_levels = levels[i:i+block_size, j:j+block_size]
            for k in range(blurness_level):
                mask = (block_levels == k)
                result[i:i+block_size, j:j+block_size][mask] = blurred[k][i:i+block_size, j:j+block_size][mask]

    # Define the center and the maximum radius
    center = (image_array.shape[1] / 2, image_array.shape[0] / 2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)
    # Decrease color saturation based on the distance from the center
    image_output = decrease_saturation_based_on_distance(result, distances, max_radius)
    
    # Convert the array back to an image
    # return Image.fromarray(result.astype(np.uint8))
    return Image.fromarray(image_output)



if __name__ == "__main__":
    image_path = r"./img_test.png"
    image = Image.open(image_path)
    image = image.convert('RGB')
    blurness_level = 10

    # Precompute the distances and levels only once 避免在处理每个图像时都重新计算距离和级别矩阵，从而提高效率。
    distances, levels = precompute_distance_and_levels(shape=np.array(image).shape, num_levels=blurness_level)

    # Measure the time before the function call
    start_time = time.time()

    result = simulate_fovea(image, distances, levels, blurness_level = blurness_level)

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
