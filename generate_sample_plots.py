#!/usr/bin/env python3
"""
生成示例对比图表，测试可视化功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_sample_data():
    """生成示例训练数据"""
    epochs = 50
    
    # ResNet18 数据
    resnet18_train_acc = 0.5 + 0.45 * (1 - np.exp(-np.linspace(0, 3, epochs))) + np.random.normal(0, 0.02, epochs)
    resnet18_test_acc = 0.45 + 0.4 * (1 - np.exp(-np.linspace(0, 2.5, epochs))) + np.random.normal(0, 0.025, epochs)
    resnet18_train_loss = 2.0 * np.exp(-np.linspace(0, 2.5, epochs)) + 0.1 + np.random.normal(0, 0.05, epochs)
    resnet18_test_loss = 2.2 * np.exp(-np.linspace(0, 2.2, epochs)) + 0.15 + np.random.normal(0, 0.06, epochs)
    
    # Self-Attention ResNet18 数据 (稍微更好的性能)
    sa_resnet18_train_acc = 0.52 + 0.47 * (1 - np.exp(-np.linspace(0, 3.2, epochs))) + np.random.normal(0, 0.02, epochs)
    sa_resnet18_test_acc = 0.47 + 0.42 * (1 - np.exp(-np.linspace(0, 2.7, epochs))) + np.random.normal(0, 0.025, epochs)
    sa_resnet18_train_loss = 1.9 * np.exp(-np.linspace(0, 2.7, epochs)) + 0.08 + np.random.normal(0, 0.05, epochs)
    sa_resnet18_test_loss = 2.1 * np.exp(-np.linspace(0, 2.4, epochs)) + 0.12 + np.random.normal(0, 0.06, epochs)
    
    # 确保数值在合理范围内
    resnet18_train_acc = np.clip(resnet18_train_acc, 0, 1)
    resnet18_test_acc = np.clip(resnet18_test_acc, 0, 1)
    sa_resnet18_train_acc = np.clip(sa_resnet18_train_acc, 0, 1)
    sa_resnet18_test_acc = np.clip(sa_resnet18_test_acc, 0, 1)
    
    resnet18_train_loss = np.clip(resnet18_train_loss, 0.05, 3)
    resnet18_test_loss = np.clip(resnet18_test_loss, 0.05, 3)
    sa_resnet18_train_loss = np.clip(sa_resnet18_train_loss, 0.05, 3)
    sa_resnet18_test_loss = np.clip(sa_resnet18_test_loss, 0.05, 3)
    
    return {
        'resnet18': {
            'train_acc': resnet18_train_acc,
            'test_acc': resnet18_test_acc,
            'train_loss': resnet18_train_loss,
            'test_loss': resnet18_test_loss
        },
        'selfattentionresnet18': {
            'train_acc': sa_resnet18_train_acc,
            'test_acc': sa_resnet18_test_acc,
            'train_loss': sa_resnet18_train_loss,
            'test_loss': sa_resnet18_test_loss
        }
    }

def generate_sample_comparison_plots():
    """生成示例对比图表"""
    # 创建输出目录
    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    
    # 生成示例数据
    sample_data = generate_sample_data()
    epochs = len(sample_data['resnet18']['train_acc'])
    
    # 创建图表
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 模型在不同测试环境下的准确率对比
    ax1 = axes[0, 0]
    models = ['resnet18', 'selfattentionresnet18']
    test_envs = [0, 1, 2]
    
    # 模拟最终测试准确率
    resnet18_final_accs = [0.85, 0.82, 0.79]
    sa_resnet18_final_accs = [0.87, 0.84, 0.81]
    
    x = np.arange(len(test_envs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, resnet18_final_accs, width, label='ResNet18', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, sa_resnet18_final_accs, width, label='Self-Attention ResNet18', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('测试环境 (Test Environment)')
    ax1.set_ylabel('测试准确率 (Test Accuracy)')
    ax1.set_title('模型在不同测试环境下的准确率 (ColoredMNIST)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"环境 {env}" for env in test_envs])
    ax1.legend(title="模型")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 2. 训练准确率曲线
    ax2 = axes[0, 1]
    epochs_range = range(1, epochs + 1)
    
    ax2.plot(epochs_range, sample_data['resnet18']['train_acc'], 
             marker='o', linestyle='-', linewidth=2, markersize=3,
             label='ResNet18 Training Accuracy', color='blue')
    ax2.plot(epochs_range, sample_data['resnet18']['test_acc'], 
             marker='x', linestyle='--', linewidth=2, markersize=3, alpha=0.8,
             label='ResNet18 Test Accuracy', color='lightblue')
    ax2.plot(epochs_range, sample_data['selfattentionresnet18']['train_acc'], 
             marker='s', linestyle='-', linewidth=2, markersize=3,
             label='Self-Attention ResNet18 Training Accuracy', color='red')
    ax2.plot(epochs_range, sample_data['selfattentionresnet18']['test_acc'], 
             marker='^', linestyle='--', linewidth=2, markersize=3, alpha=0.8,
             label='Self-Attention ResNet18 Test Accuracy', color='lightcoral')
    
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('准确率 (Accuracy)')
    ax2.set_title('模型准确率变化趋势')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right', fontsize=8)
    
    # 3. 训练损失曲线
    ax3 = axes[1, 0]
    
    ax3.plot(epochs_range, sample_data['resnet18']['train_loss'], 
             marker='o', linestyle='-', linewidth=2, markersize=3,
             label='ResNet18 Training Loss', color='blue')
    ax3.plot(epochs_range, sample_data['resnet18']['test_loss'], 
             marker='x', linestyle='--', linewidth=2, markersize=3, alpha=0.8,
             label='ResNet18 Test Loss', color='lightblue')
    ax3.plot(epochs_range, sample_data['selfattentionresnet18']['train_loss'], 
             marker='s', linestyle='-', linewidth=2, markersize=3,
             label='Self-Attention ResNet18 Training Loss', color='red')
    ax3.plot(epochs_range, sample_data['selfattentionresnet18']['test_loss'], 
             marker='^', linestyle='--', linewidth=2, markersize=3, alpha=0.8,
             label='Self-Attention ResNet18 Test Loss', color='lightcoral')
    
    ax3.set_xlabel('训练轮次 (Epoch)')
    ax3.set_ylabel('损失 (Loss)')
    ax3.set_title('模型损失变化趋势')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right', fontsize=8)
    
    # 4. 准确率分布箱线图
    ax4 = axes[1, 1]
    
    # 模拟多个实验的结果分布
    resnet18_results = np.random.normal(0.82, 0.03, 10)  # 10次实验结果
    sa_resnet18_results = np.random.normal(0.84, 0.03, 10)
    
    resnet18_results = np.clip(resnet18_results, 0, 1)
    sa_resnet18_results = np.clip(sa_resnet18_results, 0, 1)
    
    box_data = [resnet18_results, sa_resnet18_results]
    box_labels = ['ResNet18', 'Self-Attention\nResNet18']
    
    box_plot = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    
    colors = ['skyblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax4.set_ylabel('测试准确率 (Test Accuracy)')
    ax4.set_title('模型测试准确率分布 (跨不同测试环境)')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('ResNet18 vs Self-Attention ResNet18 对比实验示例图表 (ColoredMNIST)', fontsize=16, y=0.99)
    
    # 保存图表
    plot_file = output_dir / "sample_comparison_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"示例对比图表已保存: {plot_file}")
    
    # 生成单独的训练曲线图
    generate_training_curves(sample_data, output_dir)

def generate_training_curves(sample_data, output_dir):
    """生成单独的训练曲线图"""
    epochs = len(sample_data['resnet18']['train_acc'])
    epochs_range = range(1, epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 准确率曲线
    ax1.plot(epochs_range, sample_data['resnet18']['train_acc'], 
             marker='o', linestyle='-', linewidth=2, markersize=4,
             label='ResNet18 训练准确率', color='blue')
    ax1.plot(epochs_range, sample_data['resnet18']['test_acc'], 
             marker='x', linestyle='--', linewidth=2, markersize=4, alpha=0.8,
             label='ResNet18 测试准确率', color='lightblue')
    ax1.plot(epochs_range, sample_data['selfattentionresnet18']['train_acc'], 
             marker='s', linestyle='-', linewidth=2, markersize=4,
             label='Self-Attention ResNet18 训练准确率', color='red')
    ax1.plot(epochs_range, sample_data['selfattentionresnet18']['test_acc'], 
             marker='^', linestyle='--', linewidth=2, markersize=4, alpha=0.8,
             label='Self-Attention ResNet18 测试准确率', color='lightcoral')
    
    ax1.set_xlabel('训练轮次 (Epoch)')
    ax1.set_ylabel('准确率 (Accuracy)')
    ax1.set_title('模型准确率训练曲线对比')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right')
    
    # 损失曲线
    ax2.plot(epochs_range, sample_data['resnet18']['train_loss'], 
             marker='o', linestyle='-', linewidth=2, markersize=4,
             label='ResNet18 训练损失', color='blue')
    ax2.plot(epochs_range, sample_data['resnet18']['test_loss'], 
             marker='x', linestyle='--', linewidth=2, markersize=4, alpha=0.8,
             label='ResNet18 测试损失', color='lightblue')
    ax2.plot(epochs_range, sample_data['selfattentionresnet18']['train_loss'], 
             marker='s', linestyle='-', linewidth=2, markersize=4,
             label='Self-Attention ResNet18 训练损失', color='red')
    ax2.plot(epochs_range, sample_data['selfattentionresnet18']['test_loss'], 
             marker='^', linestyle='--', linewidth=2, markersize=4, alpha=0.8,
             label='Self-Attention ResNet18 测试损失', color='lightcoral')
    
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('损失 (Loss)')
    ax2.set_title('模型损失训练曲线对比')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存训练曲线图
    curves_file = output_dir / "sample_training_curves.png"
    plt.savefig(curves_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"示例训练曲线图已保存: {curves_file}")

def print_sample_statistics():
    """打印示例统计信息"""
    print("\n" + "="*60)
    print("示例实验结果统计")
    print("="*60)
    
    print("\nResNet18:")
    print("  架构: ResNet18")
    print("  参数数量: 11,181,642")
    print("  平均测试准确率: 0.8200 ± 0.0300")
    print("  最佳测试准确率: 0.8500")
    print("  最差测试准确率: 0.7900")
    print("  平均训练时间: 245.67 秒")
    
    print("\nSelf-Attention ResNet18:")
    print("  架构: Self-Attention ResNet18")
    print("  参数数量: 11,967,166")
    print("  平均测试准确率: 0.8400 ± 0.0300")
    print("  最佳测试准确率: 0.8700")
    print("  最差测试准确率: 0.8100")
    print("  平均训练时间: 267.34 秒")
    
    print("\n对比分析:")
    print("  测试准确率差异: +0.0200")
    print("  训练时间差异: +21.67 秒")
    print("  参数数量差异: +785,524")
    print("  准确率更高的模型: selfattentionresnet18")
    print("  训练更快的模型: resnet18")

if __name__ == "__main__":
    print("生成示例对比图表...")
    generate_sample_comparison_plots()
    print_sample_statistics()
    print("\n示例图表生成完成！")
    print("文件位置:")
    print("  - examples/sample_comparison_plots.png")
    print("  - examples/sample_training_curves.png")
