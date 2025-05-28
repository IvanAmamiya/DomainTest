#!/usr/bin/env python3
"""
结果记录和可视化模块
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


class ResultsLogger:
    """结果记录器"""
    
    def __init__(self, config):
        self.config = config
        self.results_path = config['output']['results_path']
        os.makedirs(self.results_path, exist_ok=True)
        
        # CSV文件路径
        self.csv_file = os.path.join(self.results_path, 'experiment_results.csv')
        self.detailed_csv = os.path.join(self.results_path, 'detailed_results.csv')
        
        # 图表保存路径
        self.plots_dir = os.path.join(self.results_path, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def log_experiment(self, dataset_info, model_info, training_summary, train_history, test_history):
        """记录单次实验结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 准备实验结果数据
        experiment_data = {
            'timestamp': timestamp,
            'experiment_name': self.config['experiment']['name'],
            'dataset_name': dataset_info['name'],
            'test_env': self.config['dataset']['test_env'],
            'input_shape': str(dataset_info['input_shape']),
            'num_classes': dataset_info['num_classes'],
            'num_environments': dataset_info['num_environments'],
            
            # 模型配置
            'model_architecture': model_info['architecture'],
            'total_parameters': model_info['total_parameters'],
            'trainable_parameters': model_info['trainable_parameters'],
            'pretrained': self.config['model']['pretrained'],
            'dropout_rate': self.config['model']['dropout_rate'],
            
            # 训练配置
            'epochs': self.config['training']['epochs'],
            'batch_size': self.config['training']['batch_size'],
            'learning_rate': self.config['training']['learning_rate'],
            'weight_decay': self.config['training']['weight_decay'],
            
            # 训练结果
            'final_train_acc': training_summary.get('final_train_acc', 0),
            'final_test_acc': training_summary.get('final_test_acc', 0),
            'best_test_acc': training_summary.get('best_test_acc', 0),
            'final_train_loss': training_summary.get('final_train_loss', 0),
            'final_test_loss': training_summary.get('final_test_loss', 0),
            'total_training_time': training_summary.get('total_training_time', 0),
            'avg_epoch_time': training_summary.get('avg_epoch_time', 0),
            
            # 设备信息
            'device': self.config['experiment']['device']
        }
        
        # 保存到CSV
        self._save_to_csv(experiment_data)
        
        # 保存详细历史
        self._save_detailed_history(timestamp, train_history, test_history)
        
        # 生成图表
        if self.config['output']['plot_results']:
            self._generate_plots(timestamp, train_history, test_history, dataset_info)
        
        # 保存完整配置和结果
        self._save_complete_results(timestamp, experiment_data, train_history, test_history)
        
        return timestamp
    
    def _save_to_csv(self, experiment_data):
        """保存实验结果到CSV"""
        df_new = pd.DataFrame([experiment_data])
        
        if os.path.exists(self.csv_file):
            df_existing = pd.read_csv(self.csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(self.csv_file, index=False)
        print(f"实验结果已保存到: {self.csv_file}")
    
    def _save_detailed_history(self, timestamp, train_history, test_history):
        """保存详细的训练历史"""
        detailed_data = []
        
        epochs = len(train_history['accuracy'])
        for epoch in range(epochs):
            row = {
                'timestamp': timestamp,
                'epoch': epoch + 1,
                'train_loss': train_history['loss'][epoch],
                'train_accuracy': train_history['accuracy'][epoch],
                'test_loss': test_history['loss'][epoch],
                'test_accuracy': test_history['accuracy'][epoch]
            }
            
            # 添加各环境的测试结果
            for key, values in test_history.items():
                if key.startswith('test_env_') and key.endswith('_accuracy'):
                    if epoch < len(values):
                        row[key] = values[epoch]
            
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        
        if os.path.exists(self.detailed_csv):
            df_existing = pd.read_csv(self.detailed_csv)
            df_combined = pd.concat([df_existing, df_detailed], ignore_index=True)
        else:
            df_combined = df_detailed
        
        df_combined.to_csv(self.detailed_csv, index=False)
    
    def _generate_plots(self, timestamp, train_history, test_history, dataset_info):
        """生成训练曲线图"""
        plt.style.use('default')
        
        # 1. 损失和准确率曲线
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_history['accuracy']) + 1)
        
        # 训练损失
        ax1.plot(epochs, train_history['loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, test_history['loss'], 'r-', label='Test Loss', linewidth=2)
        ax1.set_title('Training and Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 训练准确率
        ax2.plot(epochs, train_history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, test_history['accuracy'], 'r-', label='Test Accuracy', linewidth=2)
        ax2.set_title('Training and Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 各环境准确率
        ax3.plot(epochs, test_history['accuracy'], 'k-', label='Overall Test', linewidth=2)
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        color_idx = 0
        for key, values in test_history.items():
            if key.startswith('test_env_') and key.endswith('_accuracy'):
                env_num = key.split('_')[2]
                ax3.plot(epochs, values, color=colors[color_idx % len(colors)], 
                        label=f'Test Env {env_num}', linewidth=2, linestyle='--')
                color_idx += 1
        ax3.set_title('Test Accuracy by Environment')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 最终结果汇总
        final_results = {
            'Train Acc': train_history['accuracy'][-1],
            'Test Acc': test_history['accuracy'][-1],
            'Best Test': max(test_history['accuracy'])
        }
        
        ax4.bar(final_results.keys(), final_results.values(), 
                color=['skyblue', 'lightcoral', 'gold'])
        ax4.set_title('Final Results Summary')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        
        # 在柱状图上显示数值
        for i, (k, v) in enumerate(final_results.items()):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 添加整体标题
        fig.suptitle(f'Training Results - {dataset_info["name"]} - {timestamp}', 
                    fontsize=16, y=0.98)
        
        # 保存图表
        plot_file = os.path.join(self.plots_dir, f'training_curves_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线图已保存到: {plot_file}")
    
    def _save_complete_results(self, timestamp, experiment_data, train_history, test_history):
        """保存完整的实验结果"""
        complete_results = {
            'timestamp': timestamp,
            'experiment_data': experiment_data,
            'config': self.config,
            'train_history': dict(train_history),
            'test_history': dict(test_history)
        }
        
        results_file = os.path.join(self.results_path, f'complete_results_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    def generate_comparison_plots(self):
        """生成实验对比图表"""
        if not os.path.exists(self.csv_file):
            print("没有找到实验结果文件，无法生成对比图表")
            return
        
        df = pd.read_csv(self.csv_file)
        
        if len(df) < 2:
            print("实验数据不足，无法生成对比图表")
            return
        
        # 设置图表风格
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 不同数据集的性能对比
        if len(df['dataset_name'].unique()) > 1:
            df_grouped = df.groupby('dataset_name').agg({
                'best_test_acc': ['mean', 'std'],
                'final_test_acc': ['mean', 'std']
            }).round(4)
            
            datasets = df_grouped.index
            best_means = df_grouped[('best_test_acc', 'mean')]
            best_stds = df_grouped[('best_test_acc', 'std')]
            
            x = np.arange(len(datasets))
            axes[0, 0].bar(x, best_means, yerr=best_stds, capsize=5, 
                          color='skyblue', alpha=0.8)
            axes[0, 0].set_title('Best Test Accuracy by Dataset')
            axes[0, 0].set_xlabel('Dataset')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(datasets, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Single Dataset\nNo Comparison Available', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # 2. 不同学习率的影响
        if len(df['learning_rate'].unique()) > 1:
            lr_performance = df.groupby('learning_rate')['best_test_acc'].mean().sort_index()
            axes[0, 1].plot(lr_performance.index, lr_performance.values, 'o-', linewidth=2, markersize=8)
            axes[0, 1].set_title('Performance vs Learning Rate')
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Best Test Accuracy')
            axes[0, 1].set_xscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Single Learning Rate\nNo Comparison Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. 训练时间 vs 性能
        axes[1, 0].scatter(df['total_training_time'], df['best_test_acc'], 
                          alpha=0.7, s=100, c=df.index, cmap='viridis')
        axes[1, 0].set_title('Training Time vs Performance')
        axes[1, 0].set_xlabel('Total Training Time (s)')
        axes[1, 0].set_ylabel('Best Test Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 模型参数数量 vs 性能
        axes[1, 1].scatter(df['total_parameters'], df['best_test_acc'], 
                          alpha=0.7, s=100, c=df.index, cmap='plasma')
        axes[1, 1].set_title('Model Parameters vs Performance')
        axes[1, 1].set_xlabel('Total Parameters')
        axes[1, 1].set_ylabel('Best Test Accuracy')
        axes[1, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存对比图表
        comparison_file = os.path.join(self.plots_dir, f'comparison_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图表已保存到: {comparison_file}")
    
    def print_summary_table(self):
        """打印实验结果汇总表"""
        if not os.path.exists(self.csv_file):
            print("没有找到实验结果文件")
            return
        
        df = pd.read_csv(self.csv_file)
        
        print("\n" + "="*80)
        print("实验结果汇总")
        print("="*80)
        
        # 选择关键列进行显示
        display_columns = [
            'timestamp', 'dataset_name', 'test_env', 'best_test_acc', 
            'final_test_acc', 'total_training_time', 'total_parameters'
        ]
        
        available_columns = [col for col in display_columns if col in df.columns]
        summary_df = df[available_columns].copy()
        
        # 格式化数值
        if 'best_test_acc' in summary_df.columns:
            summary_df['best_test_acc'] = summary_df['best_test_acc'].round(4)
        if 'final_test_acc' in summary_df.columns:
            summary_df['final_test_acc'] = summary_df['final_test_acc'].round(4)
        if 'total_training_time' in summary_df.columns:
            summary_df['total_training_time'] = summary_df['total_training_time'].round(2)
        
        print(summary_df.to_string(index=False))
        print("\n" + "="*80)
        
        # 统计信息
        if len(df) > 0:
            print(f"总实验次数: {len(df)}")
            print(f"平均最佳测试准确率: {df['best_test_acc'].mean():.4f} ± {df['best_test_acc'].std():.4f}")
            print(f"最佳实验结果: {df['best_test_acc'].max():.4f}")
            print(f"平均训练时间: {df['total_training_time'].mean():.2f}s")


def create_results_logger(config):
    """创建结果记录器"""
    return ResultsLogger(config)
