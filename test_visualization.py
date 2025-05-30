#!/usr/bin/env python3
"""
测试可视化功能
生成模拟的实验结果图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

def generate_mock_data():
    """生成模拟的实验数据"""
    np.random.seed(42)
    
    # 模拟两个模型的结果
    models = ['resnet18', 'selfattentionresnet18']
    test_envs = [0, 1, 2]
    
    # 模拟基础结果数据
    mock_results = []
    
    for model in models:
        for test_env in test_envs:
            # ResNet18 基础性能
            if model == 'resnet18':
                base_acc = 0.75 + np.random.normal(0, 0.05)
                training_time = 180 + np.random.normal(0, 20)
                total_params = 11181642
            else:  # Self-Attention ResNet18
                base_acc = 0.78 + np.random.normal(0, 0.04)  # 稍微好一点
                training_time = 220 + np.random.normal(0, 25)  # 稍微慢一点
                total_params = 11967166
            
            # 根据测试环境调整性能（模拟域泛化挑战）
            env_penalty = test_env * 0.03  # 环境越复杂，性能下降越多
            final_acc = max(0.1, base_acc - env_penalty + np.random.normal(0, 0.02))
            
            mock_results.append({
                'model_type': model,
                'test_env': test_env,
                'test_accuracy': final_acc,
                'training_time': max(60, training_time),
                'model_info': {
                    'total_parameters': total_params,
                    'architecture': model
                },
                'success': True
            })
    
    return mock_results

def generate_mock_epoch_data():
    """生成模拟的训练历史数据"""
    np.random.seed(42)
    
    epoch_data = []
    models = ['resnet18', 'selfattentionresnet18']
    test_envs = [0, 1, 2]
    num_epochs = 50  # 模拟50个epoch的数据
    
    for model in models:
        for test_env in test_envs:
            # 为每个模型生成训练曲线
            if model == 'resnet18':
                # ResNet18 的学习曲线
                train_acc_final = 0.85 + np.random.normal(0, 0.03)
                test_acc_final = 0.75 + np.random.normal(0, 0.05) - test_env * 0.03
                train_loss_final = 0.3 + np.random.normal(0, 0.05)
                test_loss_final = 0.5 + np.random.normal(0, 0.1) + test_env * 0.1
            else:
                # Self-Attention ResNet18 的学习曲线
                train_acc_final = 0.87 + np.random.normal(0, 0.02)
                test_acc_final = 0.78 + np.random.normal(0, 0.04) - test_env * 0.025
                train_loss_final = 0.25 + np.random.normal(0, 0.04)
                test_loss_final = 0.45 + np.random.normal(0, 0.08) + test_env * 0.08
            
            for epoch in range(1, num_epochs + 1):
                # 生成学习曲线
                progress = epoch / num_epochs
                
                # 训练准确率（逐渐提升，带噪声）
                train_acc = 0.1 + (train_acc_final - 0.1) * (1 - np.exp(-3 * progress)) + np.random.normal(0, 0.01)
                train_acc = max(0.05, min(0.99, train_acc))
                
                # 测试准确率（更多波动）
                test_acc = 0.1 + (test_acc_final - 0.1) * (1 - np.exp(-2.5 * progress)) + np.random.normal(0, 0.02)
                test_acc = max(0.05, min(0.95, test_acc))
                
                # 训练损失（逐渐下降）
                train_loss = 2.0 * np.exp(-2 * progress) + train_loss_final + np.random.normal(0, 0.02)
                train_loss = max(0.01, train_loss)
                
                # 测试损失（下降但有波动）
                test_loss = 2.2 * np.exp(-1.8 * progress) + test_loss_final + np.random.normal(0, 0.03)
                test_loss = max(0.01, test_loss)
                
                epoch_data.append({
                    'Model': model,
                    'Test_Env': test_env,
                    'Epoch': epoch,
                    'Train_Loss': train_loss,
                    'Train_Accuracy': train_acc,
                    'Test_Loss': test_loss,
                    'Test_Accuracy_Epoch': test_acc
                })
    
    return pd.DataFrame(epoch_data)

def create_visualization(mock_results, epoch_df):
    """创建可视化图表"""
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})
    
    # 创建DataFrame
    df_data = []
    for result in mock_results:
        df_data.append({
            'Model': result['model_type'],
            'Test_Env': result['test_env'],
            'Test_Accuracy': result['test_accuracy'],
            'Training_Time': result['training_time'],
            'Total_Parameters': result['model_info']['total_parameters'],
            'Architecture': result['model_info']['architecture']
        })
    
    df = pd.DataFrame(df_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 模型准确率 vs 测试环境
    ax1 = axes[0, 0]
    models = sorted(df['Model'].unique())
    test_envs = sorted(df['Test_Env'].unique())
    x = np.arange(len(test_envs))
    bar_width = 0.35
    
    for i, model in enumerate(models):
        accuracies = []
        for env in test_envs:
            acc = df[(df['Model'] == model) & (df['Test_Env'] == env)]['Test_Accuracy'].mean()
            accuracies.append(acc)
        
        offset = (i - 0.5) * bar_width
        bars = ax1.bar(x + offset, accuracies, bar_width, label=model, alpha=0.8)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax1.annotate(f'{acc:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('测试环境 (Test Environment)')
    ax1.set_ylabel('测试准确率 (Test Accuracy)')
    ax1.set_title('模型在不同测试环境下的准确率 (ColoredMNIST)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'环境 {env}' for env in test_envs])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # 2. 训练准确率曲线
    ax2 = axes[0, 1]
    for model in models:
        model_data = epoch_df[epoch_df['Model'] == model]
        avg_data = model_data.groupby('Epoch')[['Train_Accuracy', 'Test_Accuracy_Epoch']].mean()
        
        ax2.plot(avg_data.index, avg_data['Train_Accuracy'], 
                marker='o', linewidth=2, label=f'{model} 训练准确率', markersize=3)
        ax2.plot(avg_data.index, avg_data['Test_Accuracy_Epoch'], 
                marker='x', linewidth=2, linestyle='--', alpha=0.8,
                label=f'{model} 测试准确率', markersize=4)
    
    ax2.set_xlabel('训练轮次 (Epoch)')
    ax2.set_ylabel('准确率 (Accuracy)')
    ax2.set_title('模型准确率变化趋势')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # 3. 训练损失曲线
    ax3 = axes[1, 0]
    for model in models:
        model_data = epoch_df[epoch_df['Model'] == model]
        avg_data = model_data.groupby('Epoch')[['Train_Loss', 'Test_Loss']].mean()
        
        ax3.plot(avg_data.index, avg_data['Train_Loss'], 
                marker='s', linewidth=2, label=f'{model} 训练损失', markersize=3)
        ax3.plot(avg_data.index, avg_data['Test_Loss'], 
                marker='^', linewidth=2, linestyle='--', alpha=0.8,
                label=f'{model} 测试损失', markersize=4)
    
    ax3.set_xlabel('训练轮次 (Epoch)')
    ax3.set_ylabel('损失 (Loss)')
    ax3.set_title('模型损失变化趋势')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 准确率分布箱线图
    ax4 = axes[1, 1]
    data_for_box = []
    labels = []
    colors = ['lightblue', 'lightcoral']
    
    for i, model in enumerate(models):
        model_accs = df[df['Model'] == model]['Test_Accuracy'].values
        data_for_box.append(model_accs)
        labels.append(model)
    
    box_plot = ax4.boxplot(data_for_box, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('测试准确率 (Test Accuracy)')
    ax4.set_title('模型准确率分布 (跨不同测试环境)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.0)
    
    plt.tight_layout()
    fig.suptitle('ResNet18 vs Self-Attention ResNet18 对比实验', fontsize=16, y=0.98)
    
    # 保存图表
    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    
    plot_file = output_dir / "test_comparison_visualization.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"测试可视化图表已保存: {plot_file}")
    
    return plot_file

def print_mock_summary(mock_results):
    """打印模拟实验摘要"""
    print("\n" + "="*60)
    print("模拟实验结果摘要")
    print("="*60)
    
    df = pd.DataFrame([{
        'Model': r['model_type'],
        'Test_Accuracy': r['test_accuracy'],
        'Training_Time': r['training_time'],
        'Total_Parameters': r['model_info']['total_parameters']
    } for r in mock_results])
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        print(f"\n{model}:")
        print(f"  平均测试准确率: {model_data['Test_Accuracy'].mean():.4f} ± {model_data['Test_Accuracy'].std():.4f}")
        print(f"  平均训练时间: {model_data['Training_Time'].mean():.1f} 秒")
        print(f"  模型参数数量: {model_data['Total_Parameters'].iloc[0]:,}")
        print(f"  最佳准确率: {model_data['Test_Accuracy'].max():.4f}")
        print(f"  最差准确率: {model_data['Test_Accuracy'].min():.4f}")

def main():
    """主函数"""
    print("生成测试可视化图表...")
    
    # 生成模拟数据
    mock_results = generate_mock_data()
    epoch_df = generate_mock_epoch_data()
    
    # 创建可视化
    plot_file = create_visualization(mock_results, epoch_df)
    
    # 打印摘要
    print_mock_summary(mock_results)
    
    # 保存模拟数据
    output_dir = Path("examples")
    data_file = output_dir / "mock_experiment_data.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(mock_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n模拟数据已保存: {data_file}")
    print(f"可视化图表已保存: {plot_file}")
    print("\n现在你可以查看图表效果，确认可视化功能是否符合预期!")

if __name__ == "__main__":
    main()
