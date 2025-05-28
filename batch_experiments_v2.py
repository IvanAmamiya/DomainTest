#!/usr/bin/env python3
"""
批量实验脚本
用于运行多组对比实验
"""

import copy
import itertools
from main import run_single_experiment
from config_manager import load_config, setup_experiment
from results_logger import create_results_logger


def generate_experiment_configs(base_config):
    """生成批量实验配置"""
    batch_config = base_config['batch_experiments']
    
    if not batch_config['enabled']:
        print("批量实验未启用")
        return [base_config]
    
    # 获取实验参数组合
    datasets = batch_config['datasets']
    test_envs = batch_config['test_envs']
    learning_rates = batch_config['learning_rates']
    batch_sizes = batch_config['batch_sizes']
    
    # 生成所有组合
    param_combinations = list(itertools.product(
        datasets, test_envs, learning_rates, batch_sizes
    ))
    
    configs = []
    for i, (dataset, test_env, lr, batch_size) in enumerate(param_combinations):
        config = copy.deepcopy(base_config)
        
        # 更新参数
        config['dataset']['name'] = dataset
        config['dataset']['test_env'] = test_env
        config['training']['learning_rate'] = lr
        config['training']['batch_size'] = batch_size
        
        # 更新实验名称
        config['experiment']['name'] = f"batch_exp_{i+1:03d}_{dataset}_env{test_env}_lr{lr}_bs{batch_size}"
        
        configs.append(config)
    
    return configs


def run_batch_experiments(config_path='config.yaml'):
    """运行批量实验"""
    print("开始批量实验...")
    
    # 加载基础配置
    base_config = load_config(config_path)
    
    # 生成实验配置
    experiment_configs = generate_experiment_configs(base_config)
    
    if len(experiment_configs) == 1:
        print("只有一个实验配置，运行单次实验")
        config, device = setup_experiment(config_path)
        run_single_experiment(config)
        return
    
    print(f"总共将运行 {len(experiment_configs)} 个实验")
    
    # 运行所有实验
    results = []
    successful_experiments = 0
    
    for i, config in enumerate(experiment_configs):
        print(f"\n{'='*80}")
        print(f"实验 {i+1}/{len(experiment_configs)}: {config['experiment']['name']}")
        print(f"{'='*80}")
        
        # 设置实验环境
        try:
            # 简化设置，只更新关键配置
            from config_manager import setup_seed, setup_device, create_output_dirs, validate_config
            
            validate_config(config)
            setup_seed(config)
            device = setup_device(config)
            create_output_dirs(config)
            
            # 运行实验
            success, timestamp, best_acc = run_single_experiment(config)
            
            if success:
                successful_experiments += 1
                results.append({
                    'experiment_name': config['experiment']['name'],
                    'dataset': config['dataset']['name'],
                    'test_env': config['dataset']['test_env'],
                    'learning_rate': config['training']['learning_rate'],
                    'batch_size': config['training']['batch_size'],
                    'best_accuracy': best_acc,
                    'timestamp': timestamp,
                    'success': True
                })
                print(f"✓ 实验成功: {best_acc:.4f}")
            else:
                results.append({
                    'experiment_name': config['experiment']['name'],
                    'success': False
                })
                print("✗ 实验失败")
                
        except Exception as e:
            print(f"✗ 实验 {i+1} 发生错误: {e}")
            results.append({
                'experiment_name': config['experiment']['name'],
                'success': False,
                'error': str(e)
            })
    
    # 总结结果
    print(f"\n{'='*80}")
    print("批量实验完成!")
    print(f"{'='*80}")
    print(f"总实验数: {len(experiment_configs)}")
    print(f"成功实验: {successful_experiments}")
    print(f"失败实验: {len(experiment_configs) - successful_experiments}")
    
    # 显示成功实验的结果
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        print(f"\n成功实验结果:")
        print("-" * 80)
        print(f"{'实验名称':<30} {'数据集':<15} {'测试环境':<8} {'学习率':<10} {'批大小':<8} {'最佳准确率':<10}")
        print("-" * 80)
        
        for result in successful_results:
            print(f"{result['experiment_name']:<30} "
                  f"{result['dataset']:<15} "
                  f"{result['test_env']:<8} "
                  f"{result['learning_rate']:<10.6f} "
                  f"{result['batch_size']:<8} "
                  f"{result['best_accuracy']:<10.4f}")
        
        # 找出最佳结果
        best_result = max(successful_results, key=lambda x: x['best_accuracy'])
        print(f"\n🏆 最佳实验: {best_result['experiment_name']}")
        print(f"   准确率: {best_result['best_accuracy']:.4f}")
        print(f"   参数: 数据集={best_result['dataset']}, 环境={best_result['test_env']}, "
              f"学习率={best_result['learning_rate']}, 批大小={best_result['batch_size']}")
    
    # 生成对比图表
    if successful_experiments > 1:
        print(f"\n正在生成对比图表...")
        try:
            results_logger = create_results_logger(base_config)
            results_logger.generate_comparison_plots()
            results_logger.print_summary_table()
        except Exception as e:
            print(f"生成对比图表失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量实验脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    run_batch_experiments(args.config)


if __name__ == '__main__':
    main()
