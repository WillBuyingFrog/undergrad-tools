import torch
import argparse
import random
import math


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
    overlap_left = torch.max(fovea_left, target_left)
    overlap_top = torch.max(fovea_top, target_top)
    overlap_right = torch.min(fovea_right, target_right)
    overlap_bottom = torch.min(fovea_bottom, target_bottom)

    # 计算重叠区域的宽度和高度
    overlap_width = torch.max(overlap_right - overlap_left, torch.tensor(0.0))
    overlap_height = torch.max(overlap_bottom - overlap_top, torch.tensor(0.0))

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

def simulated_annealing(tlwhs, fovea_width, fovea_height, img_width, img_height):
    # 定义初始状态
    current_x = random.uniform(0, img_width)
    current_y = random.uniform(0, img_height)
    current_score = total_coverage(current_x, current_y, tlwhs, fovea_width, fovea_height)

    # 温度调度参数
    initial_temp = 1000
    final_temp = 1
    alpha = 0.9
    temp = initial_temp

    # 模拟退火迭代
    while temp > final_temp:
        # 随机选择新的状态（邻域函数）
        next_x = current_x + random.uniform(-10, 10)  # 调整这个步长大小以适应具体问题
        next_y = current_y + random.uniform(-10, 10)
        next_x = max(0, min(next_x, img_width))  # 保持在图像范围内
        next_y = max(0, min(next_y, img_height))

        next_score = total_coverage(next_x, next_y, tlwhs, fovea_width, fovea_height)

        # 计算接受概率
        accept_probability = math.exp((next_score - current_score) / temp)
        if next_score > current_score or random.random() < accept_probability:
            current_x, current_y = next_x, next_y
            current_score = next_score

        # 降低温度
        temp *= alpha

    return current_x, current_y


if __name__ == '__main__':

    # 创建一个argument parser
    parser = argparse.ArgumentParser(description='中央凹区域位置优化算法')
    parser.add_argument('--algo_type', default='annealing', type=str, help='优化算法类型')
    parser.add_argument('--img_width', default=800, type=int, help='图像宽度')
    parser.add_argument('--img_height', default=600, type=int, help='图像高度')

    args = parser.parse_args()
    img_width = args.img_width
    img_height = args.img_height


    # 示例: 假设有一个目标列表
    tlwhs = [
        torch.tensor([10, 10, 30, 40]),  # [top, left, width, height]
        # ... 更多目标
    ]

    # 中央凹区域的位置
    x = torch.Tensor(100)
    y = torch.Tensor(100)

    if args.algo_type == 'grad':

        # 计算总覆盖率
        total_cov = total_coverage(x, y, tlwhs, 100, 100)

        # 反向传播
        total_cov.backward()

        # 查看梯度
        print(x.grad, y.grad)
    elif args.algo_type == 'annealing':

        # 使用该算法
        x, y = simulated_annealing(tlwhs, x, y, img_width, img_height)

        print(f'Simulated annealing result x={x}, y={y}')
