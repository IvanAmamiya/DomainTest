#!/usr/bin/env python3
"""
ResNet34 vs Self-Attention ResNet34 对比实验
系统性比较两种模型在不同数据集上的性能
"""

import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import copy

from config_manager import load_config, setup_experiment
from data_loader import create_dataloader
from models import create_resnet_model, create_self_attention_resnet18, create_self_attention_resnet50, create_multihead_self_attention_resnet18, get_model_info
from trainer import DomainGeneralizationTrainer
from results_logger import create_results_logger


class ComparisonExperiment:
    """对比实验管理器"""
    
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果目录
        self.results_dir = Path(f"results/comparison_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def run_single_model_experiment(self, model_type, dataset_name, test_env):
        """运行单个模型的实验"""
        print(f"\n{'='*60}")
        print(f"运行实验: {model_type} on {dataset_name} (test_env: {test_env})")
        print(f"{'='*60}")
        
        # 准备配置
        config = copy.deepcopy(self.config)
        config['model']['type'] = model_type
        config['dataset']['name'] = dataset_name
        config['dataset']['test_env'] = test_env
        
        # Update results path in config
        config['output']['results_path'] = f"{self.results_dir}/{model_type}_{dataset_name}_env{test_env}"
        
        # 创建实验日志
        logger = create_results_logger(
            config
        )
        
        try:
            # 加载数据
            data_loader = create_dataloader(config)
            train_loaders, test_loaders = data_loader.get_dataloaders()
            dataset_info = data_loader.get_dataset_info()
            
            # 使用第一个训练环境作为训练集，第一个测试环境作为测试集
            train_loader = train_loaders[0] if train_loaders else None
            test_loader = test_loaders[0] if test_loaders else None
            
            if train_loader is None or test_loader is None:
                raise ValueError("无法获取训练或测试数据加载器")
            
            input_shape = dataset_info['input_shape']
            num_classes = dataset_info['num_classes']
            
            # 创建验证集（使用训练集的一部分）
            val_loader = train_loader  # 简化处理，实际项目中应该分割数据集
            
            # 创建模型
            if model_type == 'resnet18':
                model = create_resnet_model(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=config['model']['pretrained'],
                    model_type='resnet18'
                ).to(self.device)
            elif model_type == 'selfattentionresnet18':
                model = create_self_attention_resnet18(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=config['model']['pretrained']
                ).to(self.device)
            elif model_type == 'resnet50':
                model = create_resnet_model(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=config['model']['pretrained'],
                    model_type='resnet50'
                ).to(self.device)
            elif model_type == 'selfattentionresnet50':
                model = create_self_attention_resnet50(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=config['model']['pretrained']
                ).to(self.device)
            elif model_type == 'multiheadselfattentionresnet18':
                num_heads = config['model'].get('num_heads', 4)
                model = create_multihead_self_attention_resnet18(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=config['model']['pretrained'],
                    num_heads=num_heads
                ).to(self.device)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 获取模型信息
            model_info = get_model_info(model, model_type)
            model_info.update({
                'input_channels': input_shape[0],
                'input_size': input_shape[1] if len(input_shape) > 1 else None,
                'num_classes': num_classes
            })
            
            print(f"模型信息: {model_info}")
            
            # 创建训练器
            trainer = DomainGeneralizationTrainer(
                model=model, 
                train_loaders=train_loaders, # Pass all train_loaders
                test_loaders=test_loaders,   # Pass all test_loaders
                config=config, 
                device=self.device
            )
            
            # 记录开始时间
            start_time = time.time()
            
            # 训练模型
            # trainer.train() 方法期望的参数是 num_epochs
            # trainer.train() 返回 train_history, test_history, best_test_acc
            train_history, test_history, best_test_accuracy_from_train = trainer.train(
                num_epochs=config['training']['epochs']
            )
            
            # 测试模型
            # trainer.evaluate() 方法期望的参数是 loaders 和可选的 split_name
            # 返回值是 avg_loss, avg_accuracy, env_results
            _, test_accuracy, _ = trainer.evaluate(loaders=test_loaders) # 使用所有的 test_loaders
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 保存模型
            model_path = self.results_dir / f"best_{model_type}_{dataset_name}_env{test_env}.pth"
            # The best model is stored in trainer.model after training
            torch.save(trainer.model.state_dict(), model_path)
            
            # 收集结果
            result = {
                'model_type': model_type,
                'dataset': dataset_name,
                'test_env': test_env,
                'test_accuracy': test_accuracy,
                'training_time': training_time,
                'model_info': model_info,
                'train_history': train_history,
                'test_history': test_history,  # 添加测试历史数据
                'model_path': str(model_path),
                'success': True
            }
            
            # logger.log_final_results(result) # Incorrect method name
            # Corrected method call: log_experiment expects different arguments.
            # We need to construct the arguments as expected by log_experiment.
            
            # Construct dataset_info for the logger
            dataset_info_for_logger = {
                'name': dataset_name,
                'input_shape': input_shape,
                'num_classes': num_classes,
                'num_environments': len(train_loaders) + len(test_loaders) # Approximate or get from data_loader if available
            }
            
            # Construct training_summary for the logger
            training_summary_for_logger = {
                'final_train_acc': train_history['accuracy'][-1] if train_history and train_history['accuracy'] else 0,
                'final_test_acc': test_accuracy, # This is the overall test accuracy from trainer.evaluate
                'best_test_acc': best_test_accuracy_from_train, # This is the best test accuracy during training epochs
                'final_train_loss': train_history['loss'][-1] if train_history and train_history['loss'] else 0,
                'final_test_loss': test_history['loss'][-1] if test_history and test_history['loss'] else 0, # Assuming test_history is populated by trainer
                'total_training_time': training_time,
                'avg_epoch_time': training_time / config['training']['epochs'] if config['training']['epochs'] > 0 else 0
            }

            logger.log_experiment(
                dataset_info=dataset_info_for_logger,
                model_info=model_info,
                training_summary=training_summary_for_logger,
                train_history=train_history,
                test_history=test_history # trainer.train now returns test_history as the second value
            )
            
            print(f"实验完成! 测试准确率: {test_accuracy:.4f}")
            return result
            
        except Exception as e:
            print(f"实验失败: {str(e)}")
            result = {
                'model_type': model_type,
                'dataset': dataset_name,
                'test_env': test_env,
                'error': str(e),
                'success': False
            }
            return result
    
    def run_comparison_experiments(self, datasets=None, test_envs=None):
        """运行完整的对比实验"""
        if datasets is None:
            datasets = ['ColoredMNIST', 'TerraIncognita']
        
        if test_envs is None:
            test_envs = {
                'ColoredMNIST': [0, 1, 2],
                'TerraIncognita': [0, 1, 2, 3]
            }
        
        model_types = ['selfattentionresnet50', 'resnet50']
        
        print(f"开始对比实验:")
        print(f"模型类型: {model_types}")
        print(f"数据集: {datasets}")
        print(f"测试环境: {test_envs}")
        
        all_results = []
        
        for dataset in datasets:
            envs = test_envs.get(dataset, [0])
            for test_env in envs:
                for model_type in model_types:
                    result = self.run_single_model_experiment(model_type, dataset, test_env)
                    all_results.append(result)
        
        # 保存所有结果
        results_file = self.results_dir / "comparison_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 生成分析报告
        # Pass the raw all_results to generate_analysis_report for epoch-level data access
        self.generate_analysis_report(all_results) 
        
        return all_results
    
    def generate_analysis_report(self, all_results_raw): # Renamed parameter to reflect it's raw data
        """生成分析报告"""
        print(f"\n{'='*80}")
        print("生成分析报告...")
        print(f"{'='*80}")
        
        # 过滤成功的实验
        successful_results = [r for r in all_results_raw if r.get('success', False)]
        
        if not successful_results:
            print("没有成功的实验结果")
            return
        
        # 创建DataFrame for summary statistics (as before)
        df_data = []
        for result in successful_results:
            df_data.append({
                'Model': result['model_type'],
                'Dataset': result['dataset'],
                'Test_Env': result['test_env'],
                'Test_Accuracy': result['test_accuracy'],
                'Final_Train_Accuracy': result['train_history']['accuracy'][-1] if result.get('train_history') and result['train_history'].get('accuracy') else None,
                'Training_Time': result['training_time'],
                'Total_Parameters': result['model_info']['total_parameters'] if result.get('model_info') else None,
                'Architecture': result['model_info']['architecture'] if result.get('model_info') else None
            })
        
        df = pd.DataFrame(df_data)
        
        # 保存详细结果
        csv_file = self.results_dir / "detailed_comparison.csv"
        df.to_csv(csv_file, index=False)
        
        # 生成统计摘要
        summary = self.generate_summary_statistics(df)
        
        # 保存摘要
        summary_file = self.results_dir / "comparison_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成可视化图表, pass both the summary DataFrame and the raw results for epoch data
        self.generate_comparison_plots(df, all_results_raw) # Pass raw results here
        
        # 打印摘要
        self.print_summary(summary)
    
    def generate_summary_statistics(self, df):
        """生成统计摘要"""
        summary = {}
        
        for model in df['Model'].unique():
            model_df = df[df['Model'] == model]
            
            summary[model] = {
                'avg_test_accuracy': float(model_df['Test_Accuracy'].mean()),
                'std_test_accuracy': float(model_df['Test_Accuracy'].std()),
                'avg_training_time': float(model_df['Training_Time'].mean()),
                'total_parameters': int(model_df['Total_Parameters'].iloc[0]),
                'architecture': model_df['Architecture'].iloc[0],
                'best_test_accuracy': float(model_df['Test_Accuracy'].max()),
                'worst_test_accuracy': float(model_df['Test_Accuracy'].min()),
                'avg_train_accuracy': float(model_df['Final_Train_Accuracy'].mean()) if 'Final_Train_Accuracy' in model_df and not model_df['Final_Train_Accuracy'].isnull().all() else None,
                'std_train_accuracy': float(model_df['Final_Train_Accuracy'].std()) if 'Final_Train_Accuracy' in model_df and not model_df['Final_Train_Accuracy'].isnull().all() else None,
                'best_train_accuracy': float(model_df['Final_Train_Accuracy'].max()) if 'Final_Train_Accuracy' in model_df and not model_df['Final_Train_Accuracy'].isnull().all() else None,
                'worst_train_accuracy': float(model_df['Final_Train_Accuracy'].min()) if 'Final_Train_Accuracy' in model_df and not model_df['Final_Train_Accuracy'].isnull().all() else None,
                'experiments_count': len(model_df)
            }
        
        # 比较分析
        if len(summary) >= 2:
            models = list(summary.keys())
            model1, model2 = models[0], models[1]
            
            summary['comparison'] = {
                'accuracy_difference': summary[model2]['avg_test_accuracy'] - summary[model1]['avg_test_accuracy'] if summary[model1]['avg_test_accuracy'] is not None and summary[model2]['avg_test_accuracy'] is not None else None,
                'time_difference': summary[model2]['avg_training_time'] - summary[model1]['avg_training_time'],
                'parameter_difference': summary[model2]['total_parameters'] - summary[model1]['total_parameters'],
                'better_model_accuracy': model2 if summary[model1]['avg_test_accuracy'] is not None and summary[model2]['avg_test_accuracy'] is not None and summary[model2]['avg_test_accuracy'] > summary[model1]['avg_test_accuracy'] else model1 if summary[model1]['avg_test_accuracy'] is not None and summary[model2]['avg_test_accuracy'] is not None else "N/A",
                'faster_model': model1 if summary[model1]['avg_training_time'] < summary[model2]['avg_training_time'] else model2
            }
        
        return summary
    
    def generate_comparison_plots(self, df, all_results_raw):
        """生成对比图表"""
        plt.style.use('default')
        plt.rcParams.update({'font.size': 10})

        # 从原始结果中提取每个epoch的训练和测试历史
        epoch_data = []
        for res in all_results_raw:
            if res.get('success', False) and 'train_history' in res and 'test_history' in res:
                model_type = res['model_type']
                test_env = res['test_env']
                train_losses = res['train_history'].get('loss', [])
                train_accs = res['train_history'].get('accuracy', [])
                test_losses = res.get('test_history', {}).get('loss', [])
                test_accs = res.get('test_history', {}).get('accuracy', [])
                max_epochs = max(len(train_losses), len(train_accs), len(test_losses), len(test_accs));
                
                for epoch in range(max_epochs):
                    epoch_data.append({
                        'Model': model_type,
                        'Test_Env': test_env,
                        'Epoch': epoch + 1,
                        'Train_Loss': train_losses[epoch] if epoch < len(train_losses) else None,
                        'Train_Accuracy': train_accs[epoch] if epoch < len(train_accs) else None,
                        'Test_Loss': test_losses[epoch] if epoch < len(test_losses) else None,
                        'Test_Accuracy_Epoch': test_accs[epoch] if epoch < len(test_accs) else None
                    })
        
        epoch_df = pd.DataFrame(epoch_data)

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 模型准确率 vs 测试环境 (泛化能力)
        ax1 = axes[0, 0]
        models = sorted(df['Model'].unique())
        dataset_name_title = df['Dataset'].unique()[0] if len(df['Dataset'].unique()) == 1 else "Multiple Datasets"
        active_test_envs = sorted(df['Test_Env'].unique())
        x = np.arange(len(active_test_envs))
        num_models = len(models)
        if num_models > 0:
            total_width_for_group = 0.8
            bar_width = total_width_for_group / num_models
        else:
            bar_width = 0.8

        for i, model in enumerate(models):
            accuracies = []
            for env_val in active_test_envs:
                acc = df[(df['Model'] == model) & (df['Test_Env'] == env_val)]['Test_Accuracy'].mean()
                accuracies.append(acc if not pd.isna(acc) else 0)
            offset = (i - (num_models - 1) / 2) * bar_width
            ax1.bar(x + offset, accuracies, bar_width, label=model, alpha=0.8)
        
        ax1.set_xlabel('测试环境 (Test Environment)')
        ax1.set_ylabel('平均测试准确率 (Average Test Accuracy)')
        ax1.set_title(f'模型在不同测试环境下的准确率 ({dataset_name_title})')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"环境 {env}" for env in active_test_envs])
        ax1.legend(title="模型")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 1.05)
        for p in ax1.patches:
            ax1.annotate(f"{p.get_height():.3f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', xytext=(0, 5), 
                           textcoords='offset points', fontsize=8)

        # 2. 训练准确率曲线
        ax2 = axes[0, 1]
        
        if not epoch_df.empty:
            # Plot training accuracies
            for model_type in epoch_df['Model'].unique():
                model_epoch_df = epoch_df[epoch_df['Model'] == model_type]
                numeric_cols = ['Train_Accuracy', 'Test_Accuracy_Epoch']
                available_cols = [col for col in numeric_cols if col in model_epoch_df.columns]
                avg_epoch_df = model_epoch_df.groupby('Epoch')[available_cols].mean().reset_index()
                
                if 'Train_Accuracy' in avg_epoch_df.columns:
                    ax2.plot(avg_epoch_df['Epoch'], avg_epoch_df['Train_Accuracy'], 
                            marker='o', linestyle='-', linewidth=2, 
                            label=f'{model_type} Training Accuracy')
                if 'Test_Accuracy_Epoch' in avg_epoch_df.columns:
                    ax2.plot(avg_epoch_df['Epoch'], avg_epoch_df['Test_Accuracy_Epoch'], 
                            marker='x', linestyle='--', linewidth=2, alpha=0.8,
                            label=f'{model_type} Test Accuracy')
        
        ax2.set_xlabel('训练轮次 (Epoch)')
        ax2.set_ylabel('准确率 (Accuracy)')
        ax2.set_title('模型准确率变化趋势')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='best')

        # 3. 训练损失曲线
        ax3 = axes[1, 0]
        
        if not epoch_df.empty:
            # Plot training losses
            for model_type in epoch_df['Model'].unique():
                model_epoch_df = epoch_df[epoch_df['Model'] == model_type]
                numeric_cols = ['Train_Loss', 'Test_Loss']
                available_cols = [col for col in numeric_cols if col in model_epoch_df.columns]
                avg_epoch_df = model_epoch_df.groupby('Epoch')[available_cols].mean().reset_index()
                
                if 'Train_Loss' in avg_epoch_df.columns:
                    ax3.plot(avg_epoch_df['Epoch'], avg_epoch_df['Train_Loss'], 
                            marker='s', linestyle='-', linewidth=2,
                            label=f'{model_type} Training Loss')
                if 'Test_Loss' in avg_epoch_df.columns and not avg_epoch_df['Test_Loss'].isnull().all():
                    ax3.plot(avg_epoch_df['Epoch'], avg_epoch_df['Test_Loss'], 
                            marker='^', linestyle='--', linewidth=2, alpha=0.8,
                            label=f'{model_type} Test Loss')
        
        ax3.set_xlabel('训练轮次 (Epoch)')
        ax3.set_ylabel('损失 (Loss)')
        ax3.set_title('模型损失变化趋势')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='best')
        
        # Set appropriate y-limits for loss plot
        if not epoch_df.empty and (('Train_Loss' in epoch_df.columns and epoch_df['Train_Loss'].notna().any()) or 
                                   ('Test_Loss' in epoch_df.columns and epoch_df['Test_Loss'].notna().any())):
            min_loss = min(epoch_df['Train_Loss'].min() if 'Train_Loss' in epoch_df and epoch_df['Train_Loss'].notna().any() else float('inf'),
                          epoch_df['Test_Loss'].min() if 'Test_Loss' in epoch_df and epoch_df['Test_Loss'].notna().any() else float('inf'))
            max_loss = max(epoch_df['Train_Loss'].max() if 'Train_Loss' in epoch_df and epoch_df['Train_Loss'].notna().any() else 0,
                          epoch_df['Test_Loss'].max() if 'Test_Loss' in epoch_df and epoch_df['Test_Loss'].notna().any() else 0)
            
            if pd.notna(min_loss) and pd.notna(max_loss) and max_loss > min_loss and min_loss != float('inf'):
                ax3.set_ylim(max(0, min_loss - 0.1 * (max_loss - min_loss)), max_loss + 0.1 * (max_loss - min_loss))
            elif pd.notna(max_loss) and max_loss > 0:
                ax3.set_ylim(0, max_loss * 1.1)
            else:
                ax3.set_ylim(0, 1.0)
        else:
            ax3.text(0.5, 0.5, "无 Epoch 数据用于绘制损失曲线", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax3.transAxes)

        # 4. 准确率分布箱线图
        ax4 = axes[1, 1]
        data_for_box = []
        valid_models_for_box = []
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        for model in models:
            accuracies_model_all_envs = df[df['Model'] == model]['Test_Accuracy'].dropna().values
            if len(accuracies_model_all_envs) > 0:
                data_for_box.append(accuracies_model_all_envs)
                valid_models_for_box.append(model)
        
        if data_for_box:
            box_plot = ax4.boxplot(data_for_box, labels=valid_models_for_box, patch_artist=True, widths=0.5)
            for patch, color in zip(box_plot['boxes'], colors[:len(valid_models_for_box)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax4.set_ylabel('测试准确率 (Test Accuracy)')
            ax4.set_title('模型测试准确率分布 (跨不同测试环境)')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.set_ylim(0, 1.05)
        else:
            ax4.text(0.5, 0.5, "无数据显示", horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(f'模型对比实验综合图表 ({dataset_name_title})', fontsize=16, y=0.99)
        
        plot_file = self.results_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"对比图表已保存: {plot_file}")
    
    def print_summary(self, summary):
        """打印摘要统计"""
        print(f"\n{'='*80}")
        print("实验结果摘要")
        print(f"{'='*80}")
        
        for model, stats in summary.items():
            if model == 'comparison':
                continue
            
            print(f"\n{model}:")
            print(f"  架构: {stats['architecture']}")
            print(f"  参数数量: {stats['total_parameters']:,}")
            if stats['avg_test_accuracy'] is not None:
                print(f"  平均测试准确率: {stats['avg_test_accuracy']:.4f} ± {stats['std_test_accuracy']:.4f}")
                print(f"  最佳测试准确率: {stats['best_test_accuracy']:.4f}")
                print(f"  最差测试准确率: {stats['worst_test_accuracy']:.4f}")
            else:
                print("  测试准确率: N/A")
            if stats['avg_train_accuracy'] is not None:
                print(f"  平均训练准确率: {stats['avg_train_accuracy']:.4f} ± {stats['std_train_accuracy']:.4f}")
                print(f"  最佳训练准确率: {stats['best_train_accuracy']:.4f}")
                print(f"  最差训练准确率: {stats['worst_train_accuracy']:.4f}")
            else:
                print("  训练准确率: N/A")
            print(f"  平均训练时间: {stats['avg_training_time']:.2f} 秒")
            print(f"  实验次数: {stats['experiments_count']}")
        
        if 'comparison' in summary:
            comp = summary['comparison']
            print(f"\n对比分析:")
            if comp['accuracy_difference'] is not None:
                print(f"  测试准确率差异: {comp['accuracy_difference']:+.4f}")
            else:
                print("  测试准确率差异: N/A")
            print(f"  训练时间差异: {comp['time_difference']:+.2f} 秒")
            print(f"  参数数量差异: {comp['parameter_difference']:+,}")
            print(f"  准确率更高的模型: {comp['better_model_accuracy']}")
            print(f"  训练更快的模型: {comp['faster_model']}")


def main():
    """主函数"""
    print("开始 ResNet34 vs Self-Attention ResNet34 对比实验")
    
    # 创建实验管理器
    experiment = ComparisonExperiment()
    
    # 修改配置以进行epoch为300的实验
    experiment.config['training']['epochs'] = 300 # 恢复为300个epoch进行完整训练
    experiment.config['model']['pretrained'] = True  # ImageNetの事前学習済みモデルを用いて比較実験を行う
    
    # 支持命令行参数覆盖
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, choices=['true','false'], default=None, help='是否使用预训练权重')
    parser.add_argument('--model_type', type=str, choices=['resnet18','selfattentionresnet18','multiheadselfattentionresnet18','resnet34','selfattentionresnet34','resnet50','selfattentionresnet50'], default=None, help='模型类型')
    parser.add_argument('--dataset', type=str, default=None, help='数据集名称')
    parser.add_argument('--test_envs', type=str, default=None, help='测试环境列表, 逗号分隔')
    parser.add_argument('--num_heads', type=int, default=4, help='多头自注意力的head数（仅multiheadselfattentionresnet18有效）')
    args = parser.parse_args()

    if args.pretrained is not None:
        experiment.config['model']['pretrained'] = (args.pretrained.lower() == 'true')
    if args.model_type is not None:
        experiment.config['model']['type'] = args.model_type
    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        datasets = ['ColoredMNIST']
    if args.test_envs is not None:
        test_envs = {datasets[0]: [int(e) for e in args.test_envs.split(',')]}
    else:
        test_envs = {'ColoredMNIST': [0, 1, 2]}

    # 运行实验
    results = experiment.run_comparison_experiments(datasets, test_envs)
    
    print(f"\n实验完成! 结果保存在: {experiment.results_dir}")
    print(f"总共完成 {len([r for r in results if r.get('success', False)])} 个成功的实验")


if __name__ == "__main__":
    main()
