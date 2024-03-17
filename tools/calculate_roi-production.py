import cv2
import argparse
import numpy as np
import os

# TODO 今晚写好比较帧间突变的代码


class FrogROI:
    def __init__(self, image_width, image_height, init_image_path,
                 region_scale=0.1, pixel_change_threshold=50):
        self.image_width = image_width
        self.image_height = image_height
        self.init_image_path = init_image_path
        self.current_image = cv2.imread(init_image_path)
        # 需要持续更新的比较基准（engram）是和current_image形状相同的numpy array，初始值全部置零
        self.engram = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)
        self.engram_factor = np.array([5, 3, 2.5, 2, 1.5, 1.25, 1, 0.75, 0.5, 0.25])
        # self.engram_factor = np.array([10, 2.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1])
        self.region_scale = region_scale
        self.pixel_change_threshold = pixel_change_threshold
        # 用于计算比较基准的图片
        self.engram_images = []
    
    def store_engram_images(self, image_path):
        self.engram_images.append(cv2.imread(image_path))
        if len(self.engram_images) > 10:
            self.engram_images.pop(0)
    
    def calculate_engram(self):
        num_images = len(self.engram_images)
        factor_sum = 0.0
        for i in range(num_images):
            self.engram += self.engram_images[i] * self.engram_factor[i]
            factor_sum += self.engram_factor[i]

        self.engram = self.engram / factor_sum
    
    def conpare_engram(self, target_image,
                       visualize=False, visualize_path="results/", marker=''):

        # 契合行人目标的长宽比的区域划分？
        region_w = int(self.image_width * self.region_scale)
        region_h = int(self.image_height * self.region_scale)
        num_regions_w_h = int(1 / self.region_scale)

        threshold = region_h * region_w * self.pixel_change_threshold

        detected_regions = []

        # 目标图片和engram的差异
        diff = cv2.absdiff(self.engram, target_image)

        for i in range(num_regions_w_h):
            for j in range(num_regions_w_h):
                region_diff = diff[i*region_w:(i+1)*region_w, j*region_h:(j+1)*region_h]
                region_diff_sum = region_diff.sum()
                if region_diff_sum > threshold:
                # print(f'Region ({i}, {j}) diff sum: {region_diff_sum}')
                    # 计算这个有突变的区域的tlwh，然后加入detected_regions
                    # t: 该区域的起始x坐标。l: 该区域的起始y坐标。w: 该区域的宽度。h: 该区域的高度。
                    detected_regions.append((i*region_w, j*region_h, region_w, region_h))
        
        # 如果要可视化，就在图上画出所有探测到的区域
        if visualize:
            for region in detected_regions:
                cv2.rectangle(target_image, (region[0], region[1]), (region[0]+region[2], region[1]+region[3]), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(visualize_path, f"image_{marker}_engram.jpg"), self.engram)
            cv2.imwrite(os.path.join(visualize_path, f"image_{marker}_target.jpg"), target_image)
        return detected_regions
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', default='image', type=str, help='测试数据类型。如果是dir就代表像多目标跟踪模型里一样，按顺序计算所有帧图片的感兴趣区域')
    parser.add_argument('--data_dir', default='images/', type=str, help='测试数据存放的位置')
    parser.add_argument('--region_scale', default=0.1, type=float, help='感兴趣区域的比例系数')
    parser.add_argument('--pixel_change_threshold', default=50, type=int, help='感兴趣区域的像素变化阈值')
    parser.add_argument('--debug', default=False, type=bool, help='是否开启debug输出')

    args = parser.parse_args()
    test_type = args.test_type
    data_dir = args.data_dir
    region_scale = args.region_scale
    pixel_change_threshold = args.pixel_change_threshold
    debug = args.debug

    if test_type == 'image':
        # 从data_dir中加载十五张图片（保证data_dir中有且只有十五张图片），
        # 按照文件名升序排序，依次存入images_path中
        images_path = []

        if debug:
            print(f"Loading images from {data_dir}...")

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # 如果文件是jpg格式的图片，就加入images_path
                if file.endswith('.jpg'):
                    images_path.append(os.path.join(root, file))
        init_image_path = images_path[0]
        # 获取该张图片的宽高
        init_image = cv2.imread(init_image_path)
        image_height, image_width = init_image.shape[:2]
        if debug:
            print(f"Image shape: {image_height} x {image_width}")
        
        region_detector = FrogROI(image_height=image_height, image_width=image_width, init_image_path=init_image_path,
                                  region_scale=region_scale, pixel_change_threshold=pixel_change_threshold)
        for i in range(1, len(images_path)):
            current_image = cv2.imread(images_path[i])
            region_detector.store_engram_images(images_path[i])
            region_detector.calculate_engram()
            detected_regions = region_detector.conpare_engram(current_image.astype(np.float32), 
                                                              visualize=True, visualize_path="results/", marker=str(i))
            # print(detected_regions)