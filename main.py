#!/usr/bin/env python3
"""
主实验脚本 - 重构版本
使用配置文件统一管理参数，支持结果记录和可视化
"""

import sys
import traceback
from config_manager import setup_experiment, get_parser
from data_loader import create_dataloader
from models import create_model
from trainer import DomainGeneralizationTrainer
from results_logger import create_results_logger


def run_single_experiment(config):
    """运行单次实验"""
    print(f"\n开始实验: {config['experiment']['name']}")
    
    try:
        # 1. 加载数据
        print("正在加载数据...")
        data_loader = create_dataloader(config)
        train_loaders, test_loaders = data_loader.get_dataloaders()
        dataset_info = data_loader.get_dataset_info()
        
        print(f"数据集: {dataset_info['name']}")
        print(f"输入形状: {dataset_info['input_shape']}")
        print(f"类别数量: {dataset_info['num_classes']}")
        print(f"环境数量: {dataset_info['num_environments']}")
        print(f"测试环境: {dataset_info['test_envs']}")
        
        # 2. 创建模型
        print("正在创建模型...")
        model = create_model(config, dataset_info['num_classes'], dataset_info['input_shape'])
        
        # 获取模型信息
        from models import get_model_info
        model_type = config['model'].get('type', 'resnet34')
        model_info = get_model_info(model, model_type)
        
        print(f"模型架构: {model_info['architecture']}")
        print(f"总参数数量: {model_info['total_parameters']:,}")
        print(f"可训练参数: {model_info['trainable_parameters']:,}")
        
        # 3. 训练模型
        print("正在初始化训练器...")
        device = config['experiment']['device']
        trainer = DomainGeneralizationTrainer(
            model, train_loaders, test_loaders, config, device
        )
        
        print("开始训练...")
        train_history, test_history, best_test_acc = trainer.train(config['training']['epochs'])
        
        # 4. 记录结果
        print("正在记录结果...")
        results_logger = create_results_logger(config)
        training_summary = trainer.get_training_summary()
        
        timestamp = results_logger.log_experiment(
            dataset_info, model_info, training_summary, train_history, test_history
        )
        
        print(f"\n实验完成! 时间戳: {timestamp}")
        print(f"最佳测试准确率: {best_test_acc:.4f}")
        
        return True, timestamp, best_test_acc
        
    except Exception as e:
        print(f"实验失败: {e}")
        traceback.print_exc()
        return False, None, 0


def main():
    """主函数"""
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    try:
        # 设置实验环境
        config, device = setup_experiment(args.config, args)
        
        # 检查特殊功能
        results_logger = create_results_logger(config)
        
        if args.comparison:
            print("生成对比图表...")
            results_logger.generate_comparison_plots()
            return
        
        if args.summary:
            print("显示实验汇总...")
            results_logger.print_summary_table()
            return
        
        # 运行实验
        success, timestamp, best_acc = run_single_experiment(config)
        
        if success:
            print("\n" + "="*60)
            print("实验成功完成!")
            print(f"时间戳: {timestamp}")
            print(f"最佳测试准确率: {best_acc:.4f}")
            print(f"结果已保存到: {config['output']['results_path']}")
            
            # 显示后续操作提示
            print("\n可用的后续操作:")
            print(f"1. 查看汇总: python {sys.argv[0]} --summary")
            print(f"2. 生成对比图: python {sys.argv[0]} --comparison")
            print(f"3. 运行其他实验: python {sys.argv[0]} --dataset TerraIncognita --test_env 1")
            
        else:
            print("实验失败，请检查错误信息")
            sys.exit(1)
            
    except Exception as e:
        print(f"程序运行错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
