#!/usr/bin/env python3
"""
配置管理模块
"""

import yaml
import argparse
import os
import torch
import random
import numpy as np


def load_config(config_path='config.yaml'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")


def update_config_from_args(config, args):
    """根据命令行参数更新配置"""
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.test_env is not None:
        config['dataset']['test_env'] = args.test_env
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.device:
        config['experiment']['device'] = args.device
    if args.pretrained is not None:
        config['model']['pretrained'] = args.pretrained
    if args.exp_name:
        config['experiment']['name'] = args.exp_name
    
    return config


def setup_device(config):
    """设置计算设备"""
    device_config = config['experiment']['device']
    
    if device_config == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_config
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU")
        device = 'cpu'
    
    config['experiment']['device'] = device
    return device


def setup_seed(config):
    """设置随机种子"""
    seed = config['experiment']['seed']
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 确保结果可重现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"设置随机种子: {seed}")


def create_output_dirs(config):
    """创建输出目录"""
    dirs_to_create = [
        config['output']['model_path'],
        config['output']['results_path'],
        config['output']['log_path']
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


def validate_config(config):
    """验证配置的有效性"""
    # 验证数据集
    valid_datasets = ['ColoredMNIST', 'TerraIncognita', 'PACS', 'OfficeHome', 'VLCS', 'RotatedMNIST']
    if config['dataset']['name'] not in valid_datasets:
        raise ValueError(f"不支持的数据集: {config['dataset']['name']}")
    
    # 验证训练参数
    if config['training']['epochs'] <= 0:
        raise ValueError("epochs必须大于0")
    if config['training']['batch_size'] <= 0:
        raise ValueError("batch_size必须大于0")
    if config['training']['learning_rate'] <= 0:
        raise ValueError("learning_rate必须大于0")
    
    # 验证路径
    if not os.path.exists(config['dataset']['data_dir']):
        print(f"警告: 数据目录不存在: {config['dataset']['data_dir']}")
    
    print("配置验证通过")


def get_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(description='ResNet 领域泛化实验')
    
    # 基本参数
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str,
                       choices=['ColoredMNIST', 'TerraIncognita', 'PACS', 'OfficeHome', 'VLCS'],
                       help='数据集名称')
    parser.add_argument('--test_env', type=int,
                       help='测试环境索引')
    parser.add_argument('--epochs', type=int,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int,
                       help='批大小')
    parser.add_argument('--lr', type=float,
                       help='学习率')
    parser.add_argument('--device', type=str,
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--pretrained', action='store_true',
                       help='使用预训练权重')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                       help='不使用预训练权重')
    parser.add_argument('--exp_name', type=str,
                       help='实验名称')
    
    # 功能开关
    parser.add_argument('--no_plot', action='store_true',
                       help='不生成图表')
    parser.add_argument('--comparison', action='store_true',
                       help='生成对比图表')
    parser.add_argument('--summary', action='store_true',
                       help='显示实验汇总')
    
    parser.set_defaults(pretrained=None)
    
    return parser


def print_config(config):
    """打印配置信息"""
    print("\n" + "="*60)
    print("实验配置")
    print("="*60)
    
    def print_section(section_name, section_data, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{section_name}:")
        for key, value in section_data.items():
            if isinstance(value, dict):
                print_section(key, value, indent + 1)
            else:
                print(f"{prefix}  {key}: {value}")
    
    for section, data in config.items():
        if isinstance(data, dict):
            print_section(section, data)
        else:
            print(f"{section}: {data}")
    
    print("="*60)


def save_config(config, save_path):
    """保存配置到文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


def setup_experiment(config_path=None, args=None):
    """设置完整实验环境"""
    # 加载配置
    config = load_config(config_path or 'config.yaml')
    
    # 更新配置
    if args:
        config = update_config_from_args(config, args)
        
        # 处理图表设置
        if args.no_plot:
            config['output']['plot_results'] = False
    
    # 验证配置
    validate_config(config)
    
    # 设置环境
    setup_seed(config)
    device = setup_device(config)
    create_output_dirs(config)
    
    # 打印配置
    print_config(config)
    
    return config, device
