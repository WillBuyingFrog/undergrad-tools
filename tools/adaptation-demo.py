import cv2


REGION_SCALE = 0.1
PIXEL_CHANGE_THRESHOLD = 50


def get_all_blurred_img(img):
    # 输入是cv2.imread之后的img变量
    # 输入图片只有中间部分不是模糊的，这个函数把中间不是模糊的部分也一起模糊掉


    h, w = img.shape[:2]
    center = (h//2, w//2)

    fovea_half = (center[0] // 2, center[1] // 2)

    left_up = (center[0] - fovea_half[0], center[1] - fovea_half[1])
    right_down = (center[0] + fovea_half[0], center[1] + fovea_half[1])

    processed_foveation = cv2.GaussianBlur(img[left_up[0]:right_down[0], left_up[1]:right_down[1]], (37, 37), 0)

    img[left_up[0]:right_down[0], left_up[1]:right_down[1]] = processed_foveation

    return img

def compare_images(img1, img2):
    # 输入是cv2.imread之后，在时序上前后相邻的视频两帧,img1在前,img2在后
    # 按照REGION_SCALE指定的大小分隔图片区域，并设定阈值

    region_h = int(img1.shape[0] * REGION_SCALE)
    region_w = int(img1.shape[1] * REGION_SCALE)

    num_regions_w_h = int(1 / REGION_SCALE)

    threshold = region_h * region_w * PIXEL_CHANGE_THRESHOLD

    detected_regions = []

    # 两张图片的差异
    diff = cv2.absdiff(img1, img2)
    
    for i in range(num_regions_w_h):
        for j in range(num_regions_w_h):
            # 循环到的区域，需要加总差异值的区域为i*region_w ~ (i+1)*region_w, j*region_h ~ (j+1)*region_h
            # 这个区域的差异值
            region_diff = diff[i*region_w:(i+1)*region_w, j*region_h:(j+1)*region_h]
            region_diff_sum = region_diff.sum()
            
            if region_diff_sum > threshold:
                # print(f'Region ({i}, {j}) diff sum: {region_diff_sum}')
                detected_regions.append((i, j))
    
    return detected_regions


if __name__ == '__main__':

    demo_img1_path = 'D:\\Frog\\Thesis\\undergrad\\dataset\\MOT16\\train\\MOT16-05\\foveation\\000319.jpg'

    demo_img2_path = 'D:\\Frog\\Thesis\\undergrad\\dataset\\MOT16\\train\\MOT16-05\\foveation\\000320.jpg'

    img1 = cv2.imread(demo_img1_path)
    processed_img1 = get_all_blurred_img(img1)
    img2 = cv2.imread(demo_img2_path)
    processed_img2 = get_all_blurred_img(img2)

    res = compare_images(img1, img2)

    # res包含了一系列检测到的区域，每个区域是一个元组，元组的两个元素是区域的坐标
    # 先将区域的坐标转化成原始图片中的坐标
    # 然后在原始图片中画出这些区域
    for region in res:
        left_up = (int(region[0] * img1.shape[0] * REGION_SCALE), int(region[1] * img1.shape[1] * REGION_SCALE))
        right_down = (int((region[0] + 1) * img1.shape[0] * REGION_SCALE), int((region[1] + 1) * img1.shape[1] * REGION_SCALE))
        processed_img1 = cv2.rectangle(processed_img1, left_up, right_down, (255, 0, 0), 2)
    
    cv2.imshow('img1', processed_img1)
    cv2.waitKey(0)

    print(res)
