#!/usr/bin/env python3
"""
测试混合精度训练功能
"""

import torch
import torch.nn as nn
from config_manager import load_config
from models import create_resnet_model
from trainer import DomainGeneralizationTrainer


def test_mixed_precision():
    """测试混合精度训练是否正常工作"""
    print("="*60)
    print("测试混合精度训练功能")
    print("="*60)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查AMP可用性
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("✓ torch.cuda.amp 可用")
        amp_available = True
    except ImportError:
        print("✗ torch.cuda.amp 不可用")
        amp_available = False
    
    # 加载配置
    config = load_config('config.yaml')
    print(f"混合精度配置: {config['training'].get('mixed_precision', False)}")
    
    # 创建简单的测试模型
    print("\n创建测试模型...")
    model = create_resnet_model(
        num_classes=2,
        input_channels=2, 
        pretrained=False,
        model_type='resnet34'
    )
    
    # 创建虚拟数据
    print("创建测试数据...")
    batch_size = 4
    test_x = torch.randn(batch_size, 2, 28, 28).to(device)
    test_y = torch.randint(0, 2, (batch_size,)).to(device)
    
    # 创建虚拟数据加载器
    class DummyDataLoader:
        def __init__(self, data, targets, batch_size=4):
            self.data = data
            self.targets = targets
            self.batch_size = batch_size
        
        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                yield (
                    self.data[i:i+self.batch_size],
                    self.targets[i:i+self.batch_size]
                )
        
        def __len__(self):
            return (len(self.data) + self.batch_size - 1) // self.batch_size
    
    # 创建更多测试数据
    train_data = torch.randn(20, 2, 28, 28).to(device)
    train_targets = torch.randint(0, 2, (20,)).to(device)
    test_data = torch.randn(8, 2, 28, 28).to(device)
    test_targets = torch.randint(0, 2, (8,)).to(device)
    
    train_loader = DummyDataLoader(train_data, train_targets)
    test_loader = DummyDataLoader(test_data, test_targets)
    
    # 测试混合精度训练
    print("\n测试混合精度训练...")
    trainer = DomainGeneralizationTrainer(
        model=model,
        train_loaders=[train_loader],
        test_loaders=[test_loader],
        config=config,
        device=device
    )
    
    print(f"训练器使用混合精度: {trainer.use_amp}")
    if trainer.use_amp:
        print(f"GradScaler 已初始化: {trainer.scaler is not None}")
    
    # 运行一个epoch的训练
    print("\n运行测试训练...")
    try:
        train_loss, train_acc = trainer.train_epoch()
        print(f"✓ 训练成功 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # 测试评估
        test_loss, test_acc, _ = trainer.evaluate([test_loader])
        print(f"✓ 评估成功 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        print("\n混合精度训练测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试混合精度训练的显存使用"""
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过显存测试")
        return
    
    print("\n" + "="*60)
    print("测试显存使用情况")
    print("="*60)
    
    device = torch.device('cuda')
    
    # 测试常规精度
    print("测试常规精度显存使用...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    config_fp32 = load_config('config.yaml')
    config_fp32['training']['mixed_precision'] = False
    
    model_fp32 = create_resnet_model(
        num_classes=10,
        input_channels=3,
        pretrained=False,
        model_type='resnet34'
    ).to(device)
    
    # 大批次测试
    large_batch = torch.randn(32, 3, 224, 224).to(device)
    targets = torch.randint(0, 10, (32,)).to(device)
    
    optimizer = torch.optim.SGD(model_fp32.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model_fp32.train()
    optimizer.zero_grad()
    outputs = model_fp32(large_batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    fp32_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    print(f"FP32 显存使用: {fp32_memory:.1f} MB")
    
    # 测试混合精度
    print("测试混合精度显存使用...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    config_amp = load_config('config.yaml')
    config_amp['training']['mixed_precision'] = True
    
    model_amp = create_resnet_model(
        num_classes=10,
        input_channels=3,
        pretrained=False,
        model_type='resnet34'
    ).to(device)
    
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        optimizer = torch.optim.SGD(model_amp.parameters(), lr=0.01)
        scaler = GradScaler()
        
        model_amp.train()
        optimizer.zero_grad()
        
        with autocast():
            outputs = model_amp(large_batch)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        amp_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"AMP 显存使用: {amp_memory:.1f} MB")
        
        memory_saving = fp32_memory - amp_memory
        saving_percent = (memory_saving / fp32_memory) * 100
        print(f"显存节省: {memory_saving:.1f} MB ({saving_percent:.1f}%)")
        
    except ImportError:
        print("AMP不可用，无法测试混合精度显存使用")


if __name__ == "__main__":
    success = test_mixed_precision()
    test_memory_usage()
    
    if success:
        print(f"\n{'='*60}")
        print("✓ 混合精度训练功能测试成功!")
        print("✓ 可以开始使用混合精度训练进行实验")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("✗ 混合精度训练功能测试失败")
        print("建议检查CUDA和PyTorch版本")
        print(f"{'='*60}")
