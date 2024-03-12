import cv2
import os

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
    

# 第二步：指定任意一个位置(x,y)，以此位置为中心的一个正方形区域清晰，周围模糊

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





# 测试部分

if __name__ == '__main__':

    mot_path = 'D:\\Frog\\Thesis\\undergrad\\dataset\\MOT15\\test\\KITTI-16\\img1'

    foveation_demo(mot_path)