# -*- coding: utf-8 -*-# @Time    : 2023/10/20
# @Author  : QinbuLi
# @FileName: distribute.py
import argparse
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import json
import os

class GridPlotter:
    def __init__(self, x_bins, y_bins):
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.cmap_val = plt.cm.RdYlGn_r  # Color map for visualization

    def plot(self, save_path='debug.png', val=None, show=True, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(len(self.y_bins)*0.8, len(self.x_bins)*0.8))
        ax.set_facecolor('white')
        ax.set_xticks(self.y_bins)
        ax.set_yticks(self.x_bins)
        ax.set_aspect('equal')
        ax.invert_xaxis()

        if val is not None:
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 100  # 最大红色 最小绿色
            n_colors = 5
            color_bins = np.linspace(vmin, vmax, n_colors + 1)
        for i in range(len(self.y_bins)-1):
            for j in range(len(self.x_bins)-1):
                center_x = (self.y_bins[i] + self.y_bins[i+1]) / 2
                center_y = (self.x_bins[j] + self.x_bins[j+1]) / 2
                if val is not None:
                    value = val[j, i]
                    color_idx = np.digitize(value, color_bins, right=True) - 1
                    color_idx = np.clip(color_idx, 0, n_colors - 1)
                    if 'mAP' in save_path:
                        face_color = self.cmap_val(1 - color_idx / (n_colors - 1))
                    else:
                        face_color = self.cmap_val(color_idx / (n_colors - 1))
                    text = f'{val[j, i]:.2f}'
                else:
                    text = f'({i},{j})'
                    face_color = np.random.rand(3,)
                rect = plt.Rectangle((self.y_bins[i], self.x_bins[j]),
                                    self.y_bins[i+1]-self.y_bins[i],
                                    self.x_bins[j+1]-self.x_bins[j],
                                    facecolor=face_color, edgecolor='black')
                ax.add_patch(rect)
                text_obj = ax.text(center_x, center_y, text, ha='center', va='center',
                                   fontsize=8, color='black')
                text_obj.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'),
                                           path_effects.Normal()])
        ax.set_xlabel('Y Bins')
        ax.set_ylabel('X Bins')
        ax.set_title(save_path.split('.')[0])
        plt.grid(True, lw=0.3)
        plt.tight_layout()

        # Add color bar if values are provided
        if val is not None:
            if 'mAP' in save_path or 'falseDet' in save_path or 'missing' in save_path:
                unit = '%'
            elif 'aoe' in save_path:
                unit = '°'
            elif 'ate' in save_path:
                unit = 'm'
            else:
                unit = ''
            if 'mAP' in save_path:
                sm = plt.cm.ScalarMappable(cmap=self.cmap_val, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_cmap(self.cmap_val.reversed())
            else:
                sm = plt.cm.ScalarMappable(cmap=self.cmap_val, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label(f'Value ({unit})', rotation=270, labelpad=15)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

# Example usage of GridPlotter
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid Plotter for Metrics')
    parser.add_argument('--json_file', type=str, default='/pcpt/pcpt/project/liqinbu/B2-2025-08-08/metric_first_secondmacro/metric_v1.json', help='Path to the JSON file containing metrics')
    parser.add_argument('--save_dir', type=str, default='./B2-2025-4lidar/', help='Directory to save the plots')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    x_bins = [-80, -50, -30, -10, 0, 10, 30, 50, 80]
    y_bins = [-50, -30, -10, 0, 10, 30, 50]
    n_x = len(x_bins) - 1
    n_y = len(y_bins) - 1
    plotter = GridPlotter(x_bins, y_bins)
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data_summary = data['summary']
        data_details = data['details']
        mAP        = np.zeros([n_x, n_y])  # Initialize mAP array
        falseDet   = np.zeros([n_x, n_y])  # Initialize false detection array
        missing    = np.zeros([n_x, n_y])  # Initialize missing detection array
        aoe_tp95   = np.zeros([n_x, n_y])  # Initialize AOE TP95 array
        aoe_mean   = np.zeros([n_x, n_y])  # Initialize AOE mean array
        
        ate_x_tp95 = np.zeros([n_x, n_y])  # Initialize ATE Y TP95 array
        ate_x_mean = np.zeros([n_x, n_y])  # Initialize ATE Y mean array
        ate_y_tp95 = np.zeros([n_x, n_y])  # Initialize ATE Y TP95 array
        ate_y_mean = np.zeros([n_x, n_y])  # Initialize ATE Y mean array
        ate_x_tp95 = np.zeros([n_x, n_y])  # Initialize ATE X TP95 array
        ate_x_mean = np.zeros([n_x, n_y])  # Initialize ATE X mean array
        # Fill the arrays with data from JSON
        for i_x in range(n_x):
            for i_y in range(n_y):
                mAP[i_x, i_y] = data_summary[f'mAP@10@{i_x}@{i_y}']
                falseDet[i_x, i_y] = data_summary[f'FalseDet@10@{i_x}@{i_y}']
                missing[i_x, i_y] = data_summary[f'Missing@10@{i_x}@{i_y}']
                
                aoe_tp95[i_x, i_y] = data_details[f'aoe_tp95_{i_x}_{i_y}']
                aoe_mean[i_x, i_y] = data_details[f'aoe_{i_x}_{i_y}']
                
                ate_x_mean[i_x, i_y] = data_details[f'ate_x_{i_x}_{i_y}']
                ate_x_tp95[i_x, i_y] = data_details[f'ate_x_tp95_{i_x}_{i_y}']
                ate_y_mean[i_x, i_y] = data_details[f'ate_y_{i_x}_{i_y}']
                ate_y_tp95[i_x, i_y] = data_details[f'ate_y_tp95_{i_x}_{i_y}']
                ate_x_mean[i_x, i_y] = data_details[f'ate_x_{i_x}_{i_y}']
                ate_x_tp95[i_x, i_y] = data_details[f'ate_x_tp95_{i_x}_{i_y}']
                
        plotter.plot(val=mAP, save_path=f'{args.save_dir}mAP.png', vmin=0, vmax=100)
        plotter.plot(val=falseDet, save_path=f'{args.save_dir}falseDet.png', vmin=0, vmax=100)
        plotter.plot(val=missing, save_path=f'{args.save_dir}missing.png', vmin=0, vmax=100)
        
        plotter.plot(val=aoe_tp95, save_path=f'{args.save_dir}aoe_tp95.png', vmin=0, vmax=90)
        plotter.plot(val=aoe_mean, save_path=f'{args.save_dir}aoe_mean.png', vmin=0, vmax=90)
        plotter.plot(val=ate_y_mean, save_path=f'{args.save_dir}ate_y_mean.png', vmin=0, vmax=2)
        plotter.plot(val=ate_y_tp95, save_path=f'{args.save_dir}ate_y_tp95.png', vmin=0, vmax=2)
        plotter.plot(val=ate_x_mean, save_path=f'{args.save_dir}ate_x_mean.png', vmin=0, vmax=2)
        plotter.plot(val=ate_x_tp95, save_path=f'{args.save_dir}ate_x_tp95.png', vmin=0, vmax=2)
        # plotter.plot(save_path=f'{args.save_dir}grid_debug.png', show=False)  # Debugging plot without values