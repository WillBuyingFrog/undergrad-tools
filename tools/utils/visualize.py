import os
import matplotlib.pyplot as plt


# 将单次模拟退火的结果画成图
def visualize_ann_result(tlwhs, x, y, fovea_width, fovea_height, img_width, img_height,
                         save_path='results/', img_name='test.jpg'):
    # 设定画布大小为img_width * img_height
    plt.figure(figsize=(img_width / 100, img_height / 100))

    # 画出目标框
    for tlwh in tlwhs:
        x, y, w, h = tlwh
        plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], 'r')
    # 中央凹区域的左上角坐标为(x,y)，右下角坐标为(x+fovea_width, y+fovea_height)
    plt.plot([x, x + fovea_width, x + fovea_width, x, x], [y, y, y + fovea_height, y + fovea_height, y], 'b')

    # 保存图像到save_path/img_name
    plt.savefig(os.path.join(save_path, img_name))
    plt.close()