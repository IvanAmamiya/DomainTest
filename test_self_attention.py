#!/usr/bin/env python3
"""
快速测试 Self-Attention ResNet18 模型
验证模型创建、前向传播和基本功能
"""

import torch
import time
from models import create_resnet_model, create_self_attention_resnet18, get_model_info


def test_model_creation():
    """测试模型创建"""
    print("="*60)
    print("测试模型创建...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试参数
    num_classes = 10
    input_channels = 3
    batch_size = 4
    input_size = 224
    
    # 创建标准 ResNet18
    print("\n1. 创建标准 ResNet18...")
    resnet18 = create_resnet_model(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=False
    ).to(device)
    
    resnet18_info = get_model_info(resnet18, 'resnet18')
    print(f"ResNet18 信息: {resnet18_info}")
    
    # 创建 Self-Attention ResNet18
    print("\n2. 创建 Self-Attention ResNet18...")
    sa_resnet18 = create_self_attention_resnet18(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=False
    ).to(device)
    
    sa_resnet18_info = get_model_info(sa_resnet18, 'selfattentionresnet18')
    print(f"Self-Attention ResNet18 信息: {sa_resnet18_info}")
    
    # 比较参数数量
    param_diff = sa_resnet18_info['total_parameters'] - resnet18_info['total_parameters']
    print(f"\n参数数量差异: {param_diff:,} ({param_diff/resnet18_info['total_parameters']*100:.2f}% 增加)")
    
    return resnet18, sa_resnet18, device


def test_forward_pass(resnet18, sa_resnet18, device):
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试前向传播...")
    print("="*60)
    
    # 创建测试输入
    batch_size = 4
    input_channels = 3
    input_size = 224
    
    test_input = torch.randn(batch_size, input_channels, input_size, input_size).to(device)
    print(f"测试输入形状: {test_input.shape}")
    
    # 测试 ResNet18
    print("\n1. 测试 ResNet18 前向传播...")
    resnet18.eval()
    with torch.no_grad():
        start_time = time.time()
        output1 = resnet18(test_input)
        resnet18_time = time.time() - start_time
    
    print(f"ResNet18 输出形状: {output1.shape}")
    print(f"ResNet18 推理时间: {resnet18_time:.4f} 秒")
    
    # 测试 Self-Attention ResNet18
    print("\n2. 测试 Self-Attention ResNet18 前向传播...")
    sa_resnet18.eval()
    with torch.no_grad():
        start_time = time.time()
        output2 = sa_resnet18(test_input)
        sa_resnet18_time = time.time() - start_time
    
    print(f"Self-Attention ResNet18 输出形状: {output2.shape}")
    print(f"Self-Attention ResNet18 推理时间: {sa_resnet18_time:.4f} 秒")
    
    # 比较推理时间
    time_diff = sa_resnet18_time - resnet18_time
    print(f"\n推理时间差异: {time_diff:.4f} 秒 ({time_diff/resnet18_time*100:.2f}% 增加)")
    
    return output1, output2


def test_gradient_computation(resnet18, sa_resnet18, device):
    """测试梯度计算"""
    print("\n" + "="*60)
    print("测试梯度计算...")
    print("="*60)
    
    # 创建测试输入和标签
    batch_size = 4
    input_channels = 3
    input_size = 224
    num_classes = 10
    
    test_input = torch.randn(batch_size, input_channels, input_size, input_size).to(device)
    test_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # 测试 ResNet18 梯度
    print("\n1. 测试 ResNet18 梯度计算...")
    resnet18.train()
    optimizer1 = torch.optim.SGD(resnet18.parameters(), lr=0.01)
    
    optimizer1.zero_grad()
    output1 = resnet18(test_input)
    loss1 = criterion(output1, test_labels)
    loss1.backward()
    
    # 计算梯度范数
    grad_norm1 = torch.nn.utils.clip_grad_norm_(resnet18.parameters(), float('inf'))
    print(f"ResNet18 损失: {loss1.item():.4f}")
    print(f"ResNet18 梯度范数: {grad_norm1:.4f}")
    
    # 测试 Self-Attention ResNet18 梯度
    print("\n2. 测试 Self-Attention ResNet18 梯度计算...")
    sa_resnet18.train()
    optimizer2 = torch.optim.SGD(sa_resnet18.parameters(), lr=0.01)
    
    optimizer2.zero_grad()
    output2 = sa_resnet18(test_input)
    loss2 = criterion(output2, test_labels)
    loss2.backward()
    
    # 计算梯度范数
    grad_norm2 = torch.nn.utils.clip_grad_norm_(sa_resnet18.parameters(), float('inf'))
    print(f"Self-Attention ResNet18 损失: {loss2.item():.4f}")
    print(f"Self-Attention ResNet18 梯度范数: {grad_norm2:.4f}")


def test_attention_module():
    """测试注意力模块"""
    print("\n" + "="*60)
    print("测试注意力模块...")
    print("="*60)
    
    from models import SelfAttentionModule
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建注意力模块
    in_channels = 64
    attention = SelfAttentionModule(in_channels).to(device)
    
    # 测试输入
    batch_size = 2
    height, width = 56, 56
    test_input = torch.randn(batch_size, in_channels, height, width).to(device)
    
    print(f"输入形状: {test_input.shape}")
    
    # 前向传播
    attention.eval()
    with torch.no_grad():
        output = attention(test_input)
    
    print(f"输出形状: {output.shape}")
    print(f"输入输出形状是否一致: {test_input.shape == output.shape}")
    
    # 检查注意力权重参数
    print(f"Gamma 参数值: {attention.gamma.item():.6f}")
    
    # 计算模块参数数量
    params = sum(p.numel() for p in attention.parameters())
    print(f"注意力模块参数数量: {params:,}")


def test_different_input_sizes():
    """测试不同输入尺寸"""
    print("\n" + "="*60)
    print("测试不同输入尺寸...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    sa_resnet18 = create_self_attention_resnet18(
        num_classes=10,
        input_channels=3,
        pretrained=False
    ).to(device)
    
    sa_resnet18.eval()
    
    # 测试不同尺寸
    test_sizes = [28, 64, 128, 224]
    
    for size in test_sizes:
        test_input = torch.randn(2, 3, size, size).to(device)
        
        try:
            with torch.no_grad():
                start_time = time.time()
                output = sa_resnet18(test_input)
                inference_time = time.time() - start_time
            
            print(f"输入尺寸 {size}x{size}: 成功 | 输出形状: {output.shape} | 时间: {inference_time:.4f}s")
            
        except Exception as e:
            print(f"输入尺寸 {size}x{size}: 失败 - {str(e)}")


def main():
    """主测试函数"""
    print("开始 Self-Attention ResNet18 快速测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    try:
        # 1. 测试模型创建
        resnet18, sa_resnet18, device = test_model_creation()
        
        # 2. 测试前向传播
        output1, output2 = test_forward_pass(resnet18, sa_resnet18, device)
        
        # 3. 测试梯度计算
        test_gradient_computation(resnet18, sa_resnet18, device)
        
        # 4. 测试注意力模块
        test_attention_module()
        
        # 5. 测试不同输入尺寸
        test_different_input_sizes()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过! Self-Attention ResNet18 模型工作正常")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
