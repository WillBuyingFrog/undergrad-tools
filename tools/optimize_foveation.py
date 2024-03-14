import torch
import argparse
import random
import math
from utils.logger import Logger
from utils.visualize import visualize_ann_result

def calculate_overlap_area(x, y, tlwh, fovea_width, fovea_height):
    # 中央凹区域的左上角坐标
    fovea_left = x - fovea_width / 2
    fovea_top = y - fovea_height / 2

    # 中央凹区域的右下角坐标
    fovea_right = fovea_left + fovea_width
    fovea_bottom = fovea_top + fovea_height

    # 目标的左上角和右下角坐标
    target_left = tlwh[1]
    target_top = tlwh[0]
    target_right = target_left + tlwh[2]
    target_bottom = target_top + tlwh[3]

    # 计算重叠区域的坐标
    overlap_left = max(fovea_left, target_left)
    overlap_top = max(fovea_top, target_top)
    overlap_right = min(fovea_right, target_right)
    overlap_bottom = min(fovea_bottom, target_bottom)

    # 计算重叠区域的宽度和高度
    overlap_width = max(overlap_right - overlap_left, 0.0)
    overlap_height = max(overlap_bottom - overlap_top, 0.0)

    # 计算重叠面积
    overlap_area = overlap_width * overlap_height
    return overlap_area


def calculate_coverage(x, y, tlwh, fovea_width, fovea_height):
    overlap_area = calculate_overlap_area(x, y, tlwh, fovea_width, fovea_height)
    target_area = tlwh[2] * tlwh[3]
    coverage = overlap_area / target_area
    return coverage

def total_coverage(x, y, tlwhs, fovea_width, fovea_height):
    total = 0
    for tlwh in tlwhs:
        total += calculate_coverage(x, y, tlwh, fovea_width, fovea_height)
    return total

def simulated_annealing(tlwhs, fovea_width, fovea_height, img_width, img_height, init_x=None, init_y=None):

    logger = Logger('annealing_test.txt')

    # 定义初始状态
    if init_x is not None:
        current_x = init_x
    else:
        current_x = random.uniform(0, img_width)

    if init_y is not None:
        current_y = init_y
    else:
        current_y = random.uniform(0, img_height)
    current_score = total_coverage(current_x, current_y, tlwhs, fovea_width, fovea_height)

    # 温度调度参数
    initial_temp = 1000
    final_temp = 1
    alpha = 0.95
    temp = initial_temp

    # 记录循环轮次
    counter = 0

    # 模拟退火迭代
    while temp > final_temp:
        counter += 1

        # 随机选择新的状态（邻域函数）
        next_x = current_x + random.uniform(-10, 10)  # 调整这个步长大小以适应具体问题
        next_y = current_y + random.uniform(-10, 10)

        # 不断随机采样，直到采样得到新的合法坐标
        new_pos_eff = 10.0
        while next_x + fovea_width > img_width or next_x <= 0:
            new_pos_eff *= 1.2
            new_pos_eff = min(new_pos_eff, fovea_width)
            next_x = current_x + random.uniform(-new_pos_eff, new_pos_eff) 
        new_pos_eff = 10.0
        while next_y + fovea_height > img_height or next_y <= 0:
            new_pos_eff *= 1.2
            new_pos_eff = min(new_pos_eff, fovea_height)
            next_y = current_y + random.uniform(-new_pos_eff, new_pos_eff)

        next_score = total_coverage(next_x, next_y, tlwhs, fovea_width, fovea_height)

        # 计算接受概率
        accept_probability = math.exp((current_score - next_score) / temp)

        logger.write_log(f'Epoch {counter} - Temp: {temp:.2f}, Current score: {current_score:.2f}, Next score: {next_score:.2f}, Accept probability: {accept_probability:.2f}')

        if next_score > current_score or random.random() < accept_probability:
            logger.write_log(f'   Accept new position: x={next_x:.2f}, y={next_y:.2f}, score={next_score:.2f}')
            current_x, current_y = next_x, next_y
            current_score = next_score

        # 降低温度
        temp *= alpha
    
    logger.write_log(f'Final position: x={current_x:.2f}, y={current_y:.2f}, score={current_score:.2f}')
    logger.close()  

    return current_x, current_y


if __name__ == '__main__':

    # 创建一个argument parser
    parser = argparse.ArgumentParser(description='中央凹区域位置优化算法')
    parser.add_argument('--algo_type', default='annealing', type=str, help='优化算法类型')
    parser.add_argument('--img_width', default=800, type=int, help='图像宽度')
    parser.add_argument('--img_height', default=600, type=int, help='图像高度')
    parser.add_argument('--fovea_width', default=400, type=int, help='中央凹区域宽度')
    parser.add_argument('--fovea_height', default=300, type=int, help='中央凹区域高度')
    parser.add_argument('--exp_epochs', default=1, type=int, help='算法迭代次数')

    args = parser.parse_args()
    img_width = args.img_width
    img_height = args.img_height
    fovea_width = args.fovea_width
    fovea_height = args.fovea_height
    exp_epochs = args.exp_epochs


    # 示例: 假设有一个目标列表
    tlwhs = [
        torch.tensor([200, 200, 50, 50]),  # [top, left, width, height]
        torch.tensor([100, 100, 50, 80]),
        torch.tensor([80, 80, 50, 50]),
        torch.tensor([130, 130, 50, 50]),
        torch.tensor([180, 180, 20, 20])
        # ... 更多目标
    ]

    

    if args.algo_type == 'grad':
        # 中央凹区域的位置
        x = torch.Tensor(100)
        y = torch.Tensor(100)

        # 计算总覆盖率
        total_cov = total_coverage(x, y, tlwhs, 100, 100)

        # 反向传播
        total_cov.backward()

        # 查看梯度
        print(x.grad, y.grad)
    elif args.algo_type == 'annealing':
        
        avg_x, avg_y = 0, 0
        results = []
        for i in range(args.exp_epochs):
            x, y = simulated_annealing(tlwhs, fovea_width, fovea_height, img_width, img_height)
            avg_x += x
            avg_y += y
            results.append((x, y))
            print(f'Epoch {i+1} - Simulated annealing result x={x}, y={y}')
            visualize_ann_result(tlwhs, x, y, fovea_width, fovea_height, img_width, img_height, save_path='results/', img_name=f'test_{i}.jpg',)

        
        avg_x /= args.exp_epochs
        avg_y /= args.exp_epochs

        print(f'Average position: x={avg_x:.2f}, y={avg_y:.2f}')

        
        # print(f'Simulated annealing result x={x}, y={y}')
