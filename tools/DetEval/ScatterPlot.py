# -*- coding: utf-8 -*-# @Time    : 2023/10/20
# @Author  : QinbuLi
# @FileName: test1.py
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  

def draw_cartesian_with_points(key, x_points, y_points, value=None):
    fig, ax = plt.subplots(figsize=(10, 20))
    num_points = len(x_points)
    print(f"Number of points to plot: {num_points}")
    ax.set_xlim(min(y_points), max(y_points))
    ax.set_ylim(min(x_points), max(x_points))
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.invert_xaxis()

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1.5)  # x
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1.5)  # y
    
    ax.set_xlabel('X(m)', fontsize=12)
    ax.set_ylabel('Y(m)', fontsize=12)
    ax.set_title('distribution', fontsize=14)
    
    if key == 'false':
        ax.set_title('False Positive Distribution', fontsize=14)
        # draw red points with x y
        scatter = ax.scatter(y_points, x_points, c='red', s=16, alpha=0.8, edgecolors='red', linewidths=0.5)
    elif key == 'miss':
        ax.set_title('Missed Detection Distribution', fontsize=14)
        scatter = ax.scatter(y_points, x_points, c='red', s=16, alpha=0.8, edgecolors='red', linewidths=0.5)
    else:
        ax.set_title(f'{key} distribute', fontsize=17)
        # 将value分为5个区间
        num_bins = 5
        bins = np.linspace(value.min(), value.max(), num_bins + 1)  # 计算区间边界
        bins = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 10])
        # import pdb; pdb.set_trace()
        value_bins = np.digitize(value, bins) - 1  # 区间索引（0-4）

        # 使用从绿到红的颜色映射
        cmap = plt.cm.get_cmap('RdYlGn_r', num_bins)

        # 获取颜色
        colors = cmap(value_bins)
        edgecolors = colors  # 边缘色与主色相同

        # 绘制散点图
        scatter = ax.scatter(y_points, x_points, c=value_bins, cmap=cmap,
                            s=16, alpha=0.8, edgecolors=edgecolors, linewidths=0.5)

        # 添加颜色条并设置刻度标签
        cbar = plt.colorbar(scatter, shrink=0.5, aspect=20, pad=0.02)

        # 设置颜色条刻度为区间中点，标签显示实际值范围
        cbar.set_ticks(np.linspace(0, num_bins-1, num_bins))
        tick_labels = [f'{bins[i]:.2f} - {bins[i+1]:.2f}' for i in range(num_bins)]
        cbar.set_ticklabels(tick_labels)

        # 显示整体的最小值和最大值
        cbar.set_label(f'value (min: {value.min():.2f}, max: {value.max():.2f})', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(f'./{key}.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    with open('/pcpt/pcpt/project/liqinbu/data/bevfusion/v32_B2_second1231_3dseg_cls21_cam5lidar_intensityraw_fromt12025_42wmax50_newdata_v3/metric_v1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        det_points_info = data['det_points_info']
        gt_points_info = data['gt_points_info']
        # import pdb; pdb.set_trace()
        # draw_cartesian_with_points('miss', gt_points_info['miss'])
        # draw_cartesian_with_points('match', det_points_info['match'])

        # keys = ['false', 'miss', 'y_error', 'yaw_error']
        keys = ['y_error']
        for key in keys:
            if key == 'miss':
                draw_cartesian_with_points(key, 
                                           np.array([v['coord'][0] for v in gt_points_info[key]]),
                                           np.array([v['coord'][1] for v in gt_points_info[key]]))
            elif key == 'false':
                draw_cartesian_with_points(key, 
                                           np.array([v['coord'][0] for v in det_points_info[key]]),
                                           np.array([v['coord'][1] for v in det_points_info[key]]))
            elif key == 'y_error':
                # filter class
                save_class = 'truck'
                value = det_points_info['match']
                x = np.array([v['coord'][0] for v in value])
                y = np.array([v['coord'][1] for v in value])
                class_names = np.array([v['class'] for v in value])
                y_error = np.array([v['error'][1] for v in value])
                 # 给出y_error 大于0.2的索引，过滤x，y
                y_error_indices = np.where(np.abs(y_error) > 0.2)[0]
                class_indices = np.where(class_names == save_class)[0]
                y_error_indices = np.intersect1d(y_error_indices, class_indices)
                x = x[y_error_indices]
                y = y[y_error_indices]
                y_error = y_error[y_error_indices]
                draw_cartesian_with_points(key, x, y, y_error)
            elif key == 'yaw_error':
                value = det_points_info['match']
                x = np.array([v['coord'][0] for v in value])
                y = np.array([v['coord'][1] for v in value])
                yaw_error = np.array([v['error'][6] for v in value])
                # yaw error to degrees
                yaw_error = yaw_error * 180 / np.pi
                # 给出yaw_error 大于10的索引，过滤x，y
                yaw_error_indices = np.where(np.abs(yaw_error) > 5)[0]
                x = x[yaw_error_indices]
                y = y[yaw_error_indices]
                yaw_error = yaw_error[yaw_error_indices]
                draw_cartesian_with_points('yaw_error', x, y, yaw_error)