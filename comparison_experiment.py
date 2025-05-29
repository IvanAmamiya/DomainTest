#!/usr/bin/env python3
"""
ResNet18 vs Self-Attention ResNet18 对比实验
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
from models import create_resnet_model, create_self_attention_resnet18, get_model_info
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
                    pretrained=config['model']['pretrained']
                ).to(self.device)
            elif model_type == 'selfattentionresnet18':
                model = create_self_attention_resnet18(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=config['model']['pretrained']
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
        
        model_types = ['resnet18', 'selfattentionresnet18']
        
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
        self.generate_analysis_report(all_results)
        
        return all_results
    
    def generate_analysis_report(self, results):
        """生成分析报告"""
        print(f"\n{'='*80}")
        print("生成分析报告...")
        print(f"{'='*80}")
        
        # 过滤成功的实验
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            print("没有成功的实验结果")
            return
        
        # 创建DataFrame
        df_data = []
        for result in successful_results:
            df_data.append({
                'Model': result['model_type'],
                'Dataset': result['dataset'],
                'Test_Env': result['test_env'],
                'Test_Accuracy': result['test_accuracy'],
                'Final_Train_Accuracy': result['train_history']['accuracy'][-1] if result['train_history'] and result['train_history']['accuracy'] else None,
                'Training_Time': result['training_time'],
                'Total_Parameters': result['model_info']['total_parameters'],
                'Architecture': result['model_info']['architecture']
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
        
        # 生成可视化图表
        self.generate_comparison_plots(df)
        
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
    
    def generate_comparison_plots(self, df):
        """生成对比图表"""
        plt.style.use('default')
        # 尝试增加字体大小，使其在中文环境下更清晰
        plt.rcParams.update({'font.size': 10}) # 可以根据需要调整大小
        # 如果有中文显示问题，可能需要指定支持中文的字体
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如，使用黑体
        # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # 增加了图像尺寸
        
        # 1. 准确率对比 (修改为按测试环境对比，展示泛化能力)
        ax1 = axes[0, 0]
        models = sorted(df['Model'].unique())
        
        # 假设对比主要针对单个数据集内的不同测试环境
        # 如果有多个数据集，此图可能需要调整或针对特定数据集
        dataset_name_title = df['Dataset'].unique()[0] if len(df['Dataset'].unique()) == 1 else "Multiple Datasets"
        
        # 获取当前DataFrame中实际存在的测试环境并排序
        # 这很重要，因为并非所有模型/数据集组合都必然有所有理论上的test_env
        active_test_envs = sorted(df['Test_Env'].unique())
        
        x = np.arange(len(active_test_envs))
        num_models = len(models)
        
        # 调整条形图宽度和间距的逻辑，使其更通用
        if num_models > 0:
            total_width_for_group = 0.8 # 一个测试环境组的总宽度
            bar_width = total_width_for_group / num_models
        else:
            bar_width = 0.8 # 默认宽度

        for i, model in enumerate(models):
            accuracies = []
            for env_val in active_test_envs:
                # 获取该模型在该特定测试环境下的平均测试准确率
                acc = df[(df['Model'] == model) & (df['Test_Env'] == env_val)]['Test_Accuracy'].mean()
                accuracies.append(acc if not pd.isna(acc) else 0) # pd.isna() 更稳健
            
            # 计算每个模型条形的偏移量
            offset = (i - (num_models - 1) / 2) * bar_width
            ax1.bar(x + offset, accuracies, bar_width, label=model, alpha=0.8)
        
        ax1.set_xlabel('测试环境 (Test Environment)')
        ax1.set_ylabel('平均测试准确率 (Average Test Accuracy)')
        ax1.set_title(f'模型在不同测试环境下的准确率 ({dataset_name_title})')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"环境 {env}" for env in active_test_envs]) # 使用中文标签
        ax1.legend(title="模型")
        ax1.grid(True, linestyle='--', alpha=0.7)
        # Y轴范围可以根据实际数据调整，例如0到1
        ax1.set_ylim(0, 1.05) 
        for p in ax1.patches:
            ax1.annotate(f"{p.get_height():.3f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', xytext=(0, 5), 
                           textcoords='offset points', fontsize=8)

        # 2. 训练时间对比
        ax2 = axes[0, 1]
        # 此部分逻辑保持不变，但同样可以考虑使用 active_test_envs 或按数据集的平均时间
        # 为了简化，我们这里保持原有按数据集的平均时间，如果只有一个数据集，则只有一个x点
        datasets_for_time_plot = sorted(df['Dataset'].unique())
        x_time = np.arange(len(datasets_for_time_plot))

        if num_models > 0:
             # total_width_for_group_time = 0.8
             bar_width_time = total_width_for_group / num_models # 使用与上图一致的计算方式
        else:
            bar_width_time = 0.8

        for i, model in enumerate(models):
            times = [df[(df['Model'] == model) & (df['Dataset'] == dataset)]['Training_Time'].mean() 
                    for dataset in datasets_for_time_plot]
            times = [t if not pd.isna(t) else 0 for t in times]

            offset_time = (i - (num_models - 1) / 2) * bar_width_time
            ax2.bar(x_time + offset_time, times, bar_width_time, label=model, alpha=0.8)
        
        ax2.set_xlabel('数据集 (Dataset)')
        ax2.set_ylabel('平均训练时间 (秒) (Avg Training Time (s))')
        ax2.set_title('模型训练时间对比')
        ax2.set_xticks(x_time)
        ax2.set_xticklabels(datasets_for_time_plot)
        ax2.legend(title="模型")
        ax2.grid(True, linestyle='--', alpha=0.7)
        for p in ax2.patches:
            ax2.annotate(f"{p.get_height():.1f}s", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', xytext=(0, 5), 
                           textcoords='offset points', fontsize=8)
        
        # 3. 参数数量对比
        ax3 = axes[1, 0]
        param_counts = [df[df['Model'] == model]['Total_Parameters'].iloc[0] for model in models]
        param_counts = [pc if not pd.isna(pc) else 0 for pc in param_counts]

        colors = plt.cm.Set2(np.linspace(0, 1, len(models))) # 使用不同的颜色集
        bars = ax3.bar(models, param_counts, color=colors, alpha=0.8, width=0.5) # 调整条形宽度
        ax3.set_ylabel('总参数量 (Total Parameters)')
        ax3.set_title('模型参数数量对比')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Y轴使用科学计数法

        for bar in bars:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval, 
                     f'{int(yval):,}', 
                     va='bottom', ha='center', fontsize=8) # 在条形图顶部显示数值
        
        # 4. 准确率分布箱线图 (展示泛化稳定性的一个方面)
        ax4 = axes[1, 1]
        data_for_box = []
        valid_models_for_box = []
        for model in models:
            # 收集该模型在所有测试环境下的准确率数据点
            accuracies_model_all_envs = df[df['Model'] == model]['Test_Accuracy'].dropna().values
            if len(accuracies_model_all_envs) > 0:
                data_for_box.append(accuracies_model_all_envs)
                valid_models_for_box.append(model)
        
        if data_for_box: # 仅当有数据时绘制
            box_plot = ax4.boxplot(data_for_box, labels=valid_models_for_box, patch_artist=True, widths=0.5)
            
            for patch, color in zip(box_plot['boxes'], colors[:len(valid_models_for_box)]): # 使用与参数图一致的颜色
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax4.set_ylabel('测试准确率 (Test Accuracy)')
            ax4.set_title('模型测试准确率分布 (跨不同测试环境)')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.set_ylim(0, 1.05)
        else:
            ax4.text(0.5, 0.5, "无数据显示", horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局防止标题重叠
        fig.suptitle(f'模型对比实验综合图表 ({dataset_name_title})', fontsize=16, y=0.99)
        
        # 保存图表
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
    print("开始 ResNet18 vs Self-Attention ResNet18 对比实验")
    
    # 创建实验管理器
    experiment = ComparisonExperiment()
    
    # 修改配置以进行epoch为2的实验
    experiment.config['training']['epochs'] = 2
    
    # 配置实验
    datasets = ['ColoredMNIST']  # 可以添加更多数据集
    test_envs = {
        'ColoredMNIST': [0, 1, 2]
    }
    
    # 运行实验
    results = experiment.run_comparison_experiments(datasets, test_envs)
    
    print(f"\n实验完成! 结果保存在: {experiment.results_dir}")
    print(f"总共完成 {len([r for r in results if r.get('success', False)])} 个成功的实验")


if __name__ == "__main__":
    main()
