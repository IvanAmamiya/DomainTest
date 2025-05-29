#!/usr/bin/env python3
"""
快速对比实验 - ResNet18 vs Self-Attention ResNet18
"""

import torch
import time
import json
from datetime import datetime
from pathlib import Path

from models import create_resnet_model, create_self_attention_resnet18, get_model_info
from config_manager import load_config
from data_loader import create_dataloader


def quick_model_test(model, test_loader, device):
    """快速模型测试"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 10:  # 只测试前10个批次
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0.0


def run_quick_comparison():
    """运行快速对比实验"""
    print("开始快速对比实验: ResNet18 vs Self-Attention ResNet18")
    
    # 加载配置
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("加载数据...")
    data_loader = create_dataloader(config)
    train_loaders, test_loaders = data_loader.get_dataloaders()
    dataset_info = data_loader.get_dataset_info()
    
    test_loader = test_loaders[0] if test_loaders else train_loaders[0]
    input_shape = dataset_info['input_shape']
    num_classes = dataset_info['num_classes']
    
    print(f"数据集信息:")
    print(f"  输入形状: {input_shape}")
    print(f"  类别数量: {num_classes}")
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 测试模型列表
    models_to_test = [
        ('resnet18', 'ResNet18'),
        ('selfattentionresnet18', 'Self-Attention ResNet18')
    ]
    
    for model_type, model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"测试模型: {model_name}")
        print(f"{'='*60}")
        
        try:
            # 创建模型
            start_time = time.time()
            
            if model_type == 'resnet18':
                model = create_resnet_model(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=False  # 不使用预训练以加快速度
                ).to(device)
            elif model_type == 'selfattentionresnet18':
                model = create_self_attention_resnet18(
                    num_classes=num_classes,
                    input_channels=input_shape[0],
                    pretrained=False
                ).to(device)
            
            creation_time = time.time() - start_time
            
            # 获取模型信息
            model_info = get_model_info(model, model_type)
            model_info.update({
                'input_channels': input_shape[0],
                'num_classes': num_classes
            })
            
            print(f"模型创建时间: {creation_time:.4f} 秒")
            print(f"模型参数数量: {model_info['total_parameters']:,}")
            print(f"架构: {model_info['architecture']}")
            
            # 快速前向传播测试
            start_time = time.time()
            test_accuracy = quick_model_test(model, test_loader, device)
            inference_time = time.time() - start_time
            
            print(f"快速测试准确率: {test_accuracy:.4f}")
            print(f"推理时间: {inference_time:.4f} 秒")
            
            # 保存结果
            results[model_type] = {
                'model_name': model_name,
                'model_info': model_info,
                'creation_time': creation_time,
                'test_accuracy': test_accuracy,
                'inference_time': inference_time,
                'success': True
            }
            
        except Exception as e:
            print(f"模型测试失败: {str(e)}")
            results[model_type] = {
                'model_name': model_name,
                'error': str(e),
                'success': False
            }
    
    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"quick_comparison_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("快速对比实验总结")
    print(f"{'='*80}")
    
    for model_type, result in results.items():
        if result['success']:
            print(f"\n{result['model_name']}:")
            print(f"  参数数量: {result['model_info']['total_parameters']:,}")
            print(f"  创建时间: {result['creation_time']:.4f} 秒")
            print(f"  测试准确率: {result['test_accuracy']:.4f}")
            print(f"  推理时间: {result['inference_time']:.4f} 秒")
        else:
            print(f"\n{result['model_name']}: 测试失败 - {result['error']}")
    
    # 比较分析
    if all(results[model]['success'] for model in results):
        resnet_params = results['resnet18']['model_info']['total_parameters']
        attention_params = results['selfattentionresnet18']['model_info']['total_parameters']
        param_diff = attention_params - resnet_params
        
        resnet_acc = results['resnet18']['test_accuracy']
        attention_acc = results['selfattentionresnet18']['test_accuracy']
        acc_diff = attention_acc - resnet_acc
        
        print(f"\n对比分析:")
        print(f"  参数数量差异: {param_diff:+,} ({param_diff/resnet_params*100:+.2f}%)")
        print(f"  准确率差异: {acc_diff:+.4f}")
        
        if attention_acc > resnet_acc:
            print(f"  Self-Attention ResNet18 准确率更高")
        elif resnet_acc > attention_acc:
            print(f"  标准 ResNet18 准确率更高")
        else:
            print(f"  两个模型准确率相同")
    
    print(f"\n结果已保存到: {results_file}")
    return results


if __name__ == "__main__":
    run_quick_comparison()
