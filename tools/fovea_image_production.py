import cv2
import os
import argparse
import numpy as np

def create_foveated_mapping(image_shape, center, scale=1.0, strength=0.5):
    height, width = image_shape[:2]
    center_x, center_y = center

    # 创建网格坐标
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # 计算每个点相对于中心的坐标
    delta_x = x_grid - center_x
    delta_y = y_grid - center_y

    # 计算距离中心的距离和角度
    distance = np.sqrt(delta_x**2 + delta_y**2)
    angle = np.arctan2(delta_y, delta_x)

    # 应用非线性扭曲（例如对数映射）
    distorted_distance = np.log(distance * strength + 1) * scale

    # 计算扭曲后的坐标
    distorted_x = center_x + distorted_distance * np.cos(angle)
    distorted_y = center_y + distorted_distance * np.sin(angle)

    return distorted_x.astype(np.float32), distorted_y.astype(np.float32)

# ---
# img0向img的变换
def jde_letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


# --- 
# 原始方法
# ---


# 第一步：完成视频的“中央部分清晰，周围部分模糊”的中央凹模式处理

def foveation(img):
    # 输入是cv2.imread之后的图片

    # 图片中间部分清晰，其余部分均模糊
    h, w = img.shape[:2]
    center = (w//2, h//2)

    fovea_half = (center[0] // 2, center[1] // 2)

    left_up = (center[0] - fovea_half[0], center[1] - fovea_half[1])
    right_down = (center[0] + fovea_half[0], center[1] + fovea_half[1])
    
    processed_image = cv2.GaussianBlur(img, (37, 37), 0)
    processed_image[left_up[1]:right_down[1], left_up[0]:right_down[0]] = img[left_up[1]:right_down[1], left_up[0]:right_down[0]]

    return processed_image
    

# 第二步：指定任意一个区域（tlwh格式），该区域清晰，周围模糊

def foveation_tlwh(img, tlwh, blur_factor=37):

    left_up = (tlwh[0], tlwh[1])
    right_down = (tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])


    processed_image = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
    processed_image[left_up[1]:right_down[1], left_up[0]:right_down[0]] = img[left_up[1]:right_down[1], left_up[0]:right_down[0]]

    return processed_image


# 第三步：在第二部的基础上，根据每个位置到(x,y)的距离，调整模糊程度的大小
    

# demo函数
def foveation_demo(mot_path):
    # mot_path是一个文件夹，里面有该样本下的每一帧，以图片形式储存
    
    parent_directory = os.path.abspath(os.path.join(mot_path, os.pardir))
    foveation_directory = os.path.join(parent_directory, 'static_foveation')
    print(foveation_directory)
    counter = 0

    for roots, dirs, files in os.walk(mot_path):
        for file in files:
            img = cv2.imread(os.path.join(mot_path, file))
            img = foveation(img)
            cv2.imwrite(os.path.join(foveation_directory, file), img)
            counter += 1
        #     if counter >= 2:
        #         break
        # if counter >= 2:
        #     break
            

# 输出蛇形变换的中央凹区域
def foveation_snake(img, frame_id, fovea_width, fovea_height, blur_factor=37, mode=1):
    # 获取图像宽高
    h, w = img.shape[:2]

    step_w = (w - fovea_width) // 9
    step_h = (h - fovea_height) // 5
    
    if mode == 1:
        # 按照蛇形规则，从一帧到下一帧，在宽方向上移动step_w，并且只有在宽方向上移动到头的时候，才在高方向上移动step_h
        # 一直到高方向移动到头
        current_fovea_tl = ((frame_id % 10) * step_w, ((frame_id // 10) % 6) * step_h)
        current_fovea_br = (current_fovea_tl[0] + fovea_width, current_fovea_tl[1] + fovea_height)
        processed_image = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
        processed_image[current_fovea_tl[1]:current_fovea_br[1], current_fovea_tl[0]:current_fovea_br[0]] = img[current_fovea_tl[1]:current_fovea_br[1], current_fovea_tl[0]:current_fovea_br[0]]
        return processed_image
    elif mode == 2:
        # 按照蛇形规则，从一帧到下一帧，在高方向上移动step_h，并且只有在高方向上移动到头的时候，才在宽方向上移动step_w
        # 一直到宽方向移动到头
        current_fovea_tl = (((frame_id // 6) % 10) * step_w, (frame_id % 6) * step_h)
        current_fovea_br = (current_fovea_tl[0] + fovea_width, current_fovea_tl[1] + fovea_height)
        processed_image = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
        processed_image[current_fovea_tl[1]:current_fovea_br[1], current_fovea_tl[0]:current_fovea_br[0]] = img[current_fovea_tl[1]:current_fovea_br[1], current_fovea_tl[0]:current_fovea_br[0]]
        return processed_image

    elif mode == -1:
        current_fovea_tl = ((frame_id % 10) * step_w, ((frame_id // 10) % 6) * step_h)
        current_fovea_br = (current_fovea_tl[0] + fovea_width, current_fovea_tl[1] + fovea_height)
        return current_fovea_tl, current_fovea_br
    
    elif mode == -2:
        current_fovea_tl = (((frame_id // 6) % 10) * step_w, (frame_id % 6) * step_h)
        current_fovea_br = (current_fovea_tl[0] + fovea_width, current_fovea_tl[1] + fovea_height)
        return current_fovea_tl, current_fovea_br
    

def foveation_random(img, fovea_width, fovea_height, blur_factor=37):
    h, w = img.shape[:2]
    x = np.random.randint(0, w - fovea_width)
    y = np.random.randint(0, h - fovea_height)
    processed_image = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
    processed_image[y:y+fovea_height, x:x+fovea_width] = img[y:y+fovea_height, x:x+fovea_width]
    return processed_image

def visualize_foveal_region(img, fovea_tlwh, color=(0, 255, 0)):
    x, y, w, h = fovea_tlwh
    img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Foveation')
    parser.add_argument('--fovea_type', default='simple-gaussian', type=str, help='中央凹算法类别')
    parser.add_argument('--data_dir', default='fovea_images/', type=str, help='测试数据文件夹路径')

    args = parser.parse_args()
    fovea_type = args.fovea_type
    data_dir = args.data_dir

    if fovea_type == 'simple-gaussian':
        foveation_demo(data_dir)
    elif fovea_type == 'cv2-remap':
        
        # 用os.listdir遍历data_dir下的每个文件
        for file in os.listdir(data_dir):
            image = cv2.imread(os.path.join(data_dir, file))  
            image_shape = image.shape
            center = (image_shape[1] // 2, image_shape[0] // 2)

            # 创建映射表
            map_x, map_y = create_foveated_mapping(image_shape, center, scale=1.0, strength=0.5)

            # 应用映射表进行空间扭曲
            foveated_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

            # 显示结果
            cv2.imshow('Original Image', image)
            cv2.imshow('Foveated Image', foveated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif fovea_type == 'snake':
        # 打开一个文件，位置是results/snake_result.txt，默认清空已有内容，记录蛇形扫描函数给出的每帧的中央凹区域位置
        with open('results/snake_result.txt', 'w') as snake_log:
            # 用os.listdir遍历data_dir下的每个文件
            for file in os.listdir(data_dir):
                image = cv2.imread(os.path.join(data_dir, file))
                h, w = image.shape[:2]
                fovea_width = w // 2
                fovea_height = h // 2
                snake_log.write(f'Image shape is {h}x{w} (height x width)\n')
                break
        
            snake_log.write('frame_id,tl_x,tl_y,br_x,br_y\n')
            for i in range(180):

                current_fovea_tl, current_fovea_br = foveation_snake(image, i, fovea_width, fovea_height, blur_factor=37, mode=-2)
                snake_log.write(f'{i},{current_fovea_tl[0]},{current_fovea_tl[1]},{current_fovea_br[0]},{current_fovea_br[1]}\n')