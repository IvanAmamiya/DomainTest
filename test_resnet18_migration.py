#!/usr/bin/env python3
"""
测试ResNet34到ResNet18的迁移
验证两个模型都能正常创建和运行
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.append('.')

from models import create_resnet_model, create_self_attention_resnet18, get_model_info


def test_model_creation():
    """测试模型创建"""
    print("测试ResNet18和Self-Attention ResNet18模型创建...")
    
    # 测试参数
    num_classes = 10
    input_channels = 3
    batch_size = 8
    input_size = 32
    
    # 创建测试输入
    test_input = torch.randn(batch_size, input_channels, input_size, input_size)
    
    print(f"测试输入形状: {test_input.shape}")
    
    # 测试ResNet18
    print("\n1. 测试ResNet18...")
    try:
        resnet18 = create_resnet_model(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=True,
            model_type='resnet18'
        )
        
        # 获取模型信息
        model_info = get_model_info(resnet18, 'resnet18')
        print(f"  ✓ ResNet18创建成功")
        print(f"  - 总参数: {model_info['total_parameters']:,}")
        print(f"  - 可训练参数: {model_info['trainable_parameters']:,}")
        print(f"  - 架构: {model_info['architecture']}")
        
        # 测试前向传播
        resnet18.eval()
        with torch.no_grad():
            output = resnet18(test_input)
            print(f"  - 输出形状: {output.shape}")
            assert output.shape == (batch_size, num_classes), f"期望输出形状: ({batch_size}, {num_classes}), 实际: {output.shape}"
            assert not torch.isnan(output).any(), "输出包含NaN值"
            print(f"  ✓ ResNet18前向传播正常")
            
    except Exception as e:
        print(f"  ✗ ResNet18测试失败: {e}")
        return False
    
    # 测试Self-Attention ResNet18
    print("\n2. 测试Self-Attention ResNet18...")
    try:
        self_attention_resnet18 = create_self_attention_resnet18(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=True
        )
        
        # 获取模型信息
        model_info = get_model_info(self_attention_resnet18, 'selfattentionresnet18')
        print(f"  ✓ Self-Attention ResNet18创建成功")
        print(f"  - 总参数: {model_info['total_parameters']:,}")
        print(f"  - 可训练参数: {model_info['trainable_parameters']:,}")
        print(f"  - 架构: {model_info['architecture']}")
        
        # 测试前向传播
        self_attention_resnet18.eval()
        with torch.no_grad():
            output = self_attention_resnet18(test_input)
            print(f"  - 输出形状: {output.shape}")
            assert output.shape == (batch_size, num_classes), f"期望输出形状: ({batch_size}, {num_classes}), 实际: {output.shape}"
            assert not torch.isnan(output).any(), "输出包含NaN值"
            print(f"  ✓ Self-Attention ResNet18前向传播正常")
            
    except Exception as e:
        print(f"  ✗ Self-Attention ResNet18测试失败: {e}")
        return False
    
    print("\n3. 参数对比...")
    resnet18_params = sum(p.numel() for p in resnet18.parameters())
    self_attention_params = sum(p.numel() for p in self_attention_resnet18.parameters())
    difference = self_attention_params - resnet18_params
    
    print(f"  - ResNet18参数数量: {resnet18_params:,}")
    print(f"  - Self-Attention ResNet18参数数量: {self_attention_params:,}")
    print(f"  - 参数增加量: {difference:,} ({difference/resnet18_params*100:.1f}%)")
    
    return True


def test_comparison_experiment_import():
    """测试comparison_experiment.py的导入"""
    print("\n4. 测试comparison_experiment.py导入...")
    try:
        from comparison_experiment import ComparisonExperiment
        print("  ✓ comparison_experiment.py导入成功")
        return True
    except Exception as e:
        print(f"  ✗ comparison_experiment.py导入失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("ResNet34 → ResNet18 迁移测试")
    print("="*60)
    
    tests_passed = 0
    total_tests = 2
    
    # 测试模型创建
    if test_model_creation():
        tests_passed += 1
        print("\n✓ 模型创建测试通过")
    else:
        print("\n✗ 模型创建测试失败")
    
    # 测试导入
    if test_comparison_experiment_import():
        tests_passed += 1
        print("✓ 导入测试通过")
    else:
        print("✗ 导入测试失败")
    
    print(f"\n{'='*60}")
    print(f"测试结果: {tests_passed}/{total_tests} 通过")
    
    if tests_passed == total_tests:
        print("🎉 所有测试通过！ResNet34到ResNet18迁移成功！")
        print("\n下一步:")
        print("1. 运行 python comparison_experiment.py 开始300轮训练实验")
        print("2. 监控训练过程和GPU使用情况")
        print("3. 查看自动git提交服务状态")
        return True
    else:
        print("❌ 部分测试失败，请检查问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
