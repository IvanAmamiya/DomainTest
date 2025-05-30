#!/usr/bin/env python3
"""
快速测试Self-Attention模块修复
验证NaN问题是否已解决
"""

import torch
import torch.nn as nn
from models import SelfAttentionResNet34
import warnings
warnings.filterwarnings('ignore')

def test_self_attention_nan():
    """测试Self-Attention模块是否还会产生NaN"""
    print("🔍 测试Self-Attention NaN修复...")
    
    # 创建模型
    model = SelfAttentionResNet34(num_classes=10, input_channels=3)
    model.eval()
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输入范围: [{test_input.min().item():.3f}, {test_input.max().item():.3f}]")
    
    # 测试前向传播
    with torch.no_grad():
        try:
            output = model(test_input)
            
            print(f"输出形状: {output.shape}")
            print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # 检查NaN
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            
            if has_nan:
                print("❌ 输出包含NaN!")
                return False
            elif has_inf:
                print("❌ 输出包含Inf!")
                return False
            else:
                print("✅ 输出正常，无NaN或Inf")
                return True
                
        except Exception as e:
            print(f"❌ 前向传播出错: {e}")
            return False

def test_gradient_flow():
    """测试梯度流是否正常"""
    print("\n🔍 测试梯度流...")
    
    model = SelfAttentionResNet34(num_classes=10, input_channels=3)
    model.train()
    
    # 创建测试数据
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    y = torch.randint(0, 10, (2,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # 前向传播
        output = model(x)
        loss = criterion(output, y)
        
        print(f"Loss: {loss.item():.6f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        grad_norm = 0
        param_count = 0
        nan_grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
                param_count += 1
                
                if torch.isnan(param.grad).any():
                    print(f"❌ {name} 的梯度包含NaN")
                    nan_grad_count += 1
        
        grad_norm = grad_norm ** 0.5
        
        print(f"参数数量: {param_count}")
        print(f"总梯度范数: {grad_norm:.6f}")
        print(f"NaN梯度数量: {nan_grad_count}")
        
        if nan_grad_count == 0 and grad_norm < 100:  # 合理的梯度范数
            print("✅ 梯度流正常")
            return True
        else:
            print("❌ 梯度流异常")
            return False
            
    except Exception as e:
        print(f"❌ 梯度测试出错: {e}")
        return False

def test_attention_weights():
    """测试Attention权重的数值特性"""
    print("\n🔍 测试Attention权重...")
    
    from models import SelfAttentionModule
    
    # 测试不同通道数
    for channels in [64, 128, 256, 512]:
        print(f"\n测试 {channels} 通道:")
        
        attention = SelfAttentionModule(channels)
        attention.eval()
        
        # 创建测试输入
        h, w = 14, 14  # 典型的特征图大小
        x = torch.randn(2, channels, h, w)
        
        with torch.no_grad():
            try:
                output = attention(x)
                
                # 检查输出
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()
                
                print(f"  输入范围: [{x.min().item():.3f}, {x.max().item():.3f}]")
                print(f"  输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
                print(f"  Gamma值: {attention.gamma.item():.6f}")
                
                if has_nan:
                    print(f"  ❌ {channels}通道 - 输出包含NaN")
                elif has_inf:
                    print(f"  ❌ {channels}通道 - 输出包含Inf")
                else:
                    print(f"  ✅ {channels}通道 - 正常")
                    
            except Exception as e:
                print(f"  ❌ {channels}通道 - 出错: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("Self-Attention NaN修复验证测试")
    print("=" * 50)
    
    # 运行测试
    test1 = test_self_attention_nan()
    test2 = test_gradient_flow() 
    test_attention_weights()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("🎉 所有测试通过！NaN问题已修复")
    else:
        print("⚠️  仍存在问题，需要进一步调试")
    print("=" * 50)
