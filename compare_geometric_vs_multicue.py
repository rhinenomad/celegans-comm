#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比纯几何导航和多线索导航的结果
生成可视化图表和详细分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# 创建输出目录
out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# 加载数据
geo = pd.read_csv("results/tables/e3_nav_results_geometric_only.csv")
multi = pd.read_csv("results/tables/e3_nav_results_multicue.csv")

print("=" * 60)
print("生成对比可视化图表...")
print("=" * 60)

# ========== Figure 1: 成功率对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：成功率柱状图
geo_success_rate = geo['success'].mean() * 100
multi_success_rate = multi['success'].mean() * 100

ax1 = axes[0]
bars = ax1.bar(['Geometric\nOnly', 'Multi-cue\nNavigation'], 
                [geo_success_rate, multi_success_rate],
                color=['#2E86AB', '#A23B72'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=11)
ax1.set_title('(a) Success Rate Comparison', fontweight='bold', fontsize=12)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (bar, rate) in enumerate(zip(bars, [geo_success_rate, multi_success_rate])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    # 添加变化箭头
    if i == 1:
        change = multi_success_rate - geo_success_rate
        arrow = '↑' if change > 0 else '↓'
        color = 'green' if change > 0 else 'red'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{arrow} {abs(change):.2f}%', ha='center', va='bottom',
                 fontweight='bold', fontsize=9, color=color)

# 右图：成功/失败分布饼图
ax2 = axes[1]
geo_counts = [geo['success'].sum(), (~geo['success']).sum()]
multi_counts = [multi['success'].sum(), (~multi['success']).sum()]

# 并排显示两个饼图
colors = ['#4ECDC4', '#FF6B6B']
labels = ['Success', 'Failure']

# 左饼图（纯几何）
ax2.pie(geo_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontweight': 'bold', 'fontsize': 9},
        radius=0.4, center=(-0.3, 0))
ax2.text(-0.3, -0.6, 'Geometric\nOnly', ha='center', fontweight='bold', fontsize=10)

# 右饼图（多线索）
ax2.pie(multi_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontweight': 'bold', 'fontsize': 9},
        radius=0.4, center=(0.3, 0))
ax2.text(0.3, -0.6, 'Multi-cue\nNavigation', ha='center', fontweight='bold', fontsize=10)

ax2.set_title('(b) Success vs Failure Distribution', fontweight='bold', fontsize=12)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(out_dir / "e3_geometric_vs_multicue_success.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {out_dir / 'e3_geometric_vs_multicue_success.png'}")
plt.close()

# ========== Figure 2: 路径质量对比（Stretch分布） ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

geo_successful = geo[geo['success'] == True]
multi_successful = multi[multi['success'] == True]

# 左图：Stretch分布直方图
ax1 = axes[0]
geo_stretch = geo_successful['stretch'].values
multi_stretch = multi_successful['stretch'].values

# 过滤inf值
geo_stretch = geo_stretch[np.isfinite(geo_stretch)]
multi_stretch = multi_stretch[np.isfinite(multi_stretch)]

bins = np.linspace(1, max(max(geo_stretch), max(multi_stretch)), 30)
ax1.hist(geo_stretch, bins=bins, alpha=0.6, label='Geometric Only', 
         color='#2E86AB', edgecolor='black', linewidth=0.5)
ax1.hist(multi_stretch, bins=bins, alpha=0.6, label='Multi-cue Navigation', 
         color='#A23B72', edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Stretch Ratio', fontweight='bold', fontsize=11)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax1.set_title('(a) Stretch Distribution (Successful Paths)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# 添加统计信息
geo_median = np.median(geo_stretch)
multi_median = np.median(multi_stretch)
ax1.axvline(geo_median, color='#2E86AB', linestyle='--', linewidth=2, label=f'Geo Median: {geo_median:.2f}')
ax1.axvline(multi_median, color='#A23B72', linestyle='--', linewidth=2, label=f'Multi Median: {multi_median:.2f}')
ax1.legend(fontsize=9)

# 右图：箱线图对比
ax2 = axes[1]
data_to_plot = [geo_stretch, multi_stretch]
bp = ax2.boxplot(data_to_plot, labels=['Geometric\nOnly', 'Multi-cue\nNavigation'],
                 patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][1].set_facecolor('#A23B72')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_alpha(0.7)

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.5)

ax2.set_ylabel('Stretch Ratio', fontweight='bold', fontsize=11)
ax2.set_title('(b) Stretch Ratio Box Plot', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# 添加统计信息
ax2.text(1, geo_median, f'Median: {geo_median:.2f}', 
         ha='left', va='center', fontweight='bold', fontsize=9, color='#2E86AB')
ax2.text(2, multi_median, f'Median: {multi_median:.2f}', 
         ha='left', va='center', fontweight='bold', fontsize=9, color='#A23B72')

plt.tight_layout()
plt.savefig(out_dir / "e3_geometric_vs_multicue_stretch.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {out_dir / 'e3_geometric_vs_multicue_stretch.png'}")
plt.close()

# ========== Figure 3: 路径长度对比 ==========
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

geo_path_len = geo_successful['path_len'].values
multi_path_len = multi_successful['path_len'].values

# 创建对比柱状图
x = np.arange(2)
width = 0.35

geo_mean = np.mean(geo_path_len)
multi_mean = np.mean(multi_path_len)
geo_std = np.std(geo_path_len)
multi_std = np.std(multi_path_len)

bars1 = ax.bar(x - width/2, [geo_mean, 0], width, label='Geometric Only', 
               color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, [0, multi_mean], width, label='Multi-cue Navigation', 
               color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加误差条
ax.errorbar(x[0] - width/2, geo_mean, yerr=geo_std, fmt='none', color='black', capsize=5, linewidth=1.5)
ax.errorbar(x[1] + width/2, multi_mean, yerr=multi_std, fmt='none', color='black', capsize=5, linewidth=1.5)

ax.set_ylabel('Average Path Length (hops)', fontweight='bold', fontsize=11)
ax.set_title('Average Path Length Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['Geometric\nOnly', 'Multi-cue\nNavigation'], fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
ax.text(x[0] - width/2, geo_mean + geo_std + 0.5, f'{geo_mean:.2f}±{geo_std:.2f}',
        ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.text(x[1] + width/2, multi_mean + multi_std + 0.5, f'{multi_mean:.2f}±{multi_std:.2f}',
        ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(out_dir / "e3_geometric_vs_multicue_pathlength.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {out_dir / 'e3_geometric_vs_multicue_pathlength.png'}")
plt.close()

# ========== Figure 4: 综合对比雷达图 ==========
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# 计算指标（归一化到0-1）
categories = ['Success\nRate', 'Path\nQuality\n(1/stretch)', 'Optimal\nPath\nRatio', 'Avg Path\nLength\n(1/length)']
N = len(categories)

# 归一化函数（越大越好）
def normalize_success_rate(rate):
    return rate / 100.0

def normalize_path_quality(stretch_median):
    return 1.0 / stretch_median if stretch_median > 0 else 0

def normalize_optimal_ratio(ratio):
    return ratio / 100.0

def normalize_path_length(avg_len):
    return 1.0 / avg_len if avg_len > 0 else 0

geo_values = [
    normalize_success_rate(geo_success_rate),
    normalize_path_quality(np.median(geo_stretch)),
    normalize_optimal_ratio((geo_stretch == 1.0).sum() / len(geo_stretch) * 100),
    normalize_path_length(np.mean(geo_path_len))
]

multi_values = [
    normalize_success_rate(multi_success_rate),
    normalize_path_quality(np.median(multi_stretch)),
    normalize_optimal_ratio((multi_stretch == 1.0).sum() / len(multi_stretch) * 100),
    normalize_path_length(np.mean(multi_path_len))
]

# 计算角度
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合

geo_values += geo_values[:1]
multi_values += multi_values[:1]

# 绘制
ax.plot(angles, geo_values, 'o-', linewidth=2, label='Geometric Only', color='#2E86AB')
ax.fill(angles, geo_values, alpha=0.25, color='#2E86AB')
ax.plot(angles, multi_values, 'o-', linewidth=2, label='Multi-cue Navigation', color='#A23B72')
ax.fill(angles, multi_values, alpha=0.25, color='#A23B72')

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True)

ax.set_title('Comprehensive Comparison (Normalized Metrics)', 
             fontweight='bold', fontsize=12, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(out_dir / "e3_geometric_vs_multicue_radar.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {out_dir / 'e3_geometric_vs_multicue_radar.png'}")
plt.close()

print()
print("=" * 60)
print("所有对比图表已生成完成！")
print("=" * 60)

