import os
import matplotlib.pyplot as plt


# 将单次模拟退火的结果画成图
def visualize_ann_result(tlwhs, fovea_x, fovea_y, fovea_width, fovea_height, img_width, img_height,
                         save_path='results/', img_name='test.jpg'):
    

    # 画出目标框
    for tlwh in tlwhs:
        x, y, w, h = tlwh
        plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], 'r')
    # 中央凹区域的左上角坐标为(fovea_x,fovea_y)，右下角坐标为(fovea_x+fovea_width, fovea_y+fovea_height)
    plt.plot([fovea_x, fovea_x + fovea_width, fovea_x + fovea_width, fovea_x, fovea_x],
             [fovea_y, fovea_y, fovea_y + fovea_height, fovea_y + fovea_height, fovea_y], 'b')

    # 设置当前画的图的横轴画到img_width，纵轴最大值画到img_height
    plt.xlim(0, img_width)
    plt.ylim(0, img_height)

    # 保存图像到save_path/img_name
    plt.savefig(os.path.join(save_path, img_name))
    plt.close()