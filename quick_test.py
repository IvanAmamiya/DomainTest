#!/usr/bin/env python3
"""
快速测试脚本 - ResNet + ColoredMNIST
用于验证框架是否正常工作
"""

import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# 添加DomainBed路径
sys.path.append('./DomainBed')

def quick_test():
    print("=== ResNet + DomainBed 快速测试 ===")
    
    # 检查PyTorch和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试ResNet-18模型创建
    print("\n--- 测试ResNet-18模型 ---")
    try:
        model = models.resnet18(pretrained=False)  # 先不下载预训练权重
        print(f"ResNet-18模型创建成功")
        print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 修改分类器
        model.fc = nn.Linear(model.fc.in_features, 10)  # 10个类别用于MNIST
        
        model = model.to(device)
        print("ResNet-18模型移动到设备成功")
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"前向传播测试成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"ResNet-18模型测试失败: {e}")
        return False
    
    # 测试DomainBed数据集导入
    print("\n--- 测试DomainBed数据集 ---")
    try:
        from domainbed import datasets
        print("DomainBed导入成功")
        
        # 列出可用数据集
        print(f"可用数据集: {datasets.DATASETS}")
        
        # 测试ColoredMNIST数据集
        dataset_name = 'ColoredMNIST'
        print(f"\n测试 {dataset_name} 数据集...")
        
        # 检查数据目录
        data_dir = './DomainBed/domainbed/data/'
        if os.path.exists(data_dir):
            print(f"数据目录存在: {data_dir}")
        else:
            print(f"数据目录不存在: {data_dir}")
            print("将尝试创建...")
            os.makedirs(data_dir, exist_ok=True)
        
        # 创建数据集
        hparams = {
            'batch_size': 32,
            'data_augmentation': True,
            'resnet18': False,
            'resnet_dropout': 0.0,
            'nonlinear_classifier': False,
            'class_balanced': False
        }
        
        dataset = datasets.get_dataset_class(dataset_name)(
            data_dir, [0], hparams
        )
        
        print(f"数据集创建成功")
        print(f"输入形状: {dataset.input_shape}")
        print(f"类别数量: {dataset.num_classes}")
        print(f"环境数量: {len(dataset)}")
        print(f"环境名称: {dataset.ENVIRONMENTS}")
        
        # 测试数据加载
        from torch.utils.data import DataLoader
        first_env = dataset[0]
        loader = DataLoader(first_env, batch_size=4, shuffle=True)
        
        for batch_idx, (x, y) in enumerate(loader):
            print(f"第一个批次 - 输入形状: {x.shape}, 标签形状: {y.shape}")
            print(f"标签范围: {y.min().item()} - {y.max().item()}")
            break
        
        print("数据加载测试成功！")
        
    except Exception as e:
        print(f"DomainBed数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== 所有测试通过！可以开始正式训练 ===")
    return True

def print_coloredmnist_env_mapping():
    """打印 ColoredMNIST 的 env 编号与 domain 名称对应关系"""
    from data_loader import DomainDataLoader
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['dataset']['name'] = 'ColoredMNIST'
    loader = DomainDataLoader(config)
    env_names = loader.environment_names
    print("ColoredMNIST 环境编号与 domain 名称对应表：")
    for idx, name in enumerate(env_names):
        print(f"  env{idx}: {name}")

if __name__ == '__main__':
    print_coloredmnist_env_mapping()
    success = quick_test()
    if success:
        print("\n使用以下命令开始训练:")
        print("python3 main.py --dataset ColoredMNIST --test_env 0 --epochs 10")
        print("python3 main.py --dataset TerraIncognita --test_env 0 --epochs 20")
    else:
        print("\n测试失败，请检查环境配置")
