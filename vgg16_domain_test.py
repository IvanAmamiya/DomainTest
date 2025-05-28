#!/usr/bin/env python3
"""
DomainBed数据集 + VGG-16 领域泛化测试框架
使用外部引入的方式，独立于DomainBed的训练框架
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from collections import defaultdict
import argparse
from tqdm import tqdm

# 添加DomainBed路径以便导入数据集
sys.path.append('./DomainBed')
from domainbed import datasets

class VGG16DomainModel(nn.Module):
    """基于VGG-16的领域泛化模型，适配小尺寸图像"""
    
    def __init__(self, num_classes, input_channels=3, input_size=224, pretrained=True, dropout_rate=0.5):
        super(VGG16DomainModel, self).__init__()
        
        self.input_size = input_size
        
        # 如果输入尺寸很小（如MNIST的28x28），我们需要调整架构
        if input_size <= 32:
            # 对于小图像，使用简化的VGG架构
            self.features = nn.Sequential(
                # 第一组
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
                
                # 第二组
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
                
                # 第三组 - 不再池化，保持7x7
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                
                # 全局平均池化到固定尺寸
                nn.AdaptiveAvgPool2d((7, 7))
            )
            
            # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(256 * 7 * 7, 1024),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
            
        else:
            # 对于正常大小图像，使用标准VGG-16
            self.backbone = models.vgg16(pretrained=pretrained)
            
            # 如果输入通道数不是3，需要修改第一层
            if input_channels != 3:
                original_conv = self.backbone.features[0]
                new_conv = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
                
                if pretrained and input_channels == 2:
                    with torch.no_grad():
                        new_conv.weight[:, :2, :, :] = original_conv.weight[:, :2, :, :]
                        new_conv.bias = original_conv.bias
                
                self.backbone.features[0] = new_conv
            
            self.features = self.backbone.features
            
            # 修改分类器部分
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        # 如果是小图像，需要上采样到合适尺寸
        if self.input_size <= 32 and x.size(-1) != 28:
            x = torch.nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        elif self.input_size > 32 and x.size(-1) < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def get_features(self, x):
        """提取特征向量，用于域对抗训练等高级方法"""
        if self.input_size <= 32 and x.size(-1) != 28:
            x = torch.nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        elif self.input_size > 32 and x.size(-1) < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        features = self.features(x)
        return features.view(features.size(0), -1)

class DomainDataLoader:
    """DomainBed数据集加载器"""
    
    def __init__(self, dataset_name, data_dir, test_envs, hparams=None):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.test_envs = test_envs
        
        # 默认超参数
        if hparams is None:
            self.hparams = {
                'batch_size': 32,
                'data_augmentation': True,
                'resnet18': False,
                'resnet_dropout': 0.0,
                'nonlinear_classifier': False,
                'class_balanced': False
            }
        else:
            self.hparams = hparams
            
        # 获取数据集
        self.dataset = datasets.get_dataset_class(dataset_name)(
            data_dir, test_envs, self.hparams
        )
        
        self.input_shape = self.dataset.input_shape
        self.num_classes = self.dataset.num_classes
        
    def get_dataloaders(self, batch_size=None):
        """获取训练和测试的DataLoader"""
        if batch_size is None:
            batch_size = self.hparams['batch_size']
            
        train_loaders = []
        test_loaders = []
        
        for i, env_dataset in enumerate(self.dataset):
            loader = DataLoader(
                env_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            if i in self.test_envs:
                test_loaders.append(loader)
            else:
                train_loaders.append(loader)
                
        return train_loaders, test_loaders
    
    def get_environment_names(self):
        """获取环境名称"""
        return self.dataset.ENVIRONMENTS

class DomainGeneralizationTrainer:
    """领域泛化训练器"""
    
    def __init__(self, model, train_loaders, test_loaders, device='cuda'):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.device = device
        self.model.to(device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # 训练历史
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 合并所有训练环境的数据
        all_batches = []
        for loader in self.train_loaders:
            for batch in loader:
                all_batches.append(batch)
        
        # 随机打乱批次
        np.random.shuffle(all_batches)
        
        for x, y in tqdm(all_batches, desc="Training"):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
            
        avg_loss = total_loss / len(all_batches)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, loaders, split_name=""):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        env_results = []
        
        with torch.no_grad():
            for env_idx, loader in enumerate(loaders):
                env_loss = 0
                env_correct = 0
                env_samples = 0
                
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    
                    env_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    env_correct += (predicted == y).sum().item()
                    env_samples += y.size(0)
                
                env_acc = env_correct / env_samples if env_samples > 0 else 0
                env_results.append({
                    'env': env_idx,
                    'accuracy': env_acc,
                    'loss': env_loss / len(loader) if len(loader) > 0 else 0
                })
                
                total_loss += env_loss
                total_correct += env_correct
                total_samples += env_samples
        
        avg_loss = total_loss / sum(len(loader) for loader in loaders) if loaders else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_accuracy, env_results
    
    def train(self, num_epochs):
        """完整训练过程"""
        print(f"开始训练 {num_epochs} 个epochs...")
        print(f"使用设备: {self.device}")
        print(f"训练环境数量: {len(self.train_loaders)}")
        print(f"测试环境数量: {len(self.test_loaders)}")
        
        best_test_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            
            # 评估训练集
            train_eval_loss, train_eval_acc, train_env_results = self.evaluate(
                self.train_loaders, "train"
            )
            
            # 评估测试集
            test_loss, test_acc, test_env_results = self.evaluate(
                self.test_loaders, "test"
            )
            self.test_history['loss'].append(test_loss)
            self.test_history['accuracy'].append(test_acc)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印结果
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            
            # 打印各环境的详细结果
            print("各环境测试结果:")
            for result in test_env_results:
                print(f"  环境 {result['env']}: Acc: {result['accuracy']:.4f}")
            
            # 保存最好的模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), 'best_vgg16_domain_model.pth')
                print(f"保存最佳模型 (测试准确率: {best_test_acc:.4f})")
        
        print(f"\n训练完成！最佳测试准确率: {best_test_acc:.4f}")
        return self.train_history, self.test_history

def main():
    parser = argparse.ArgumentParser(description='VGG-16 领域泛化测试')
    parser.add_argument('--dataset', type=str, default='TerraIncognita',
                       choices=['TerraIncognita', 'PACS', 'OfficeHome', 'VLCS', 'ColoredMNIST'],
                       help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./DomainBed/domainbed/data/',
                       help='数据目录')
    parser.add_argument('--test_env', type=int, default=0,
                       help='测试环境索引')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (cuda/cpu/auto)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='使用预训练权重')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    print(f"数据集: {args.dataset}")
    print(f"测试环境: {args.test_env}")
    
    # 加载数据
    hparams = {
        'batch_size': args.batch_size,
        'data_augmentation': True,
        'resnet18': False,
        'resnet_dropout': 0.0,
        'nonlinear_classifier': False,
        'class_balanced': False
    }
    
    try:
        data_loader = DomainDataLoader(
            args.dataset, 
            args.data_dir, 
            [args.test_env], 
            hparams
        )
        
        train_loaders, test_loaders = data_loader.get_dataloaders(args.batch_size)
        
        print(f"输入形状: {data_loader.input_shape}")
        print(f"类别数量: {data_loader.num_classes}")
        print(f"环境名称: {data_loader.get_environment_names()}")
        
        # 创建模型
        input_size = max(data_loader.input_shape[1], data_loader.input_shape[2])  # 获取输入图像的尺寸
        model = VGG16DomainModel(
            num_classes=data_loader.num_classes,
            input_channels=data_loader.input_shape[0],  # 从数据集获取输入通道数
            input_size=input_size,  # 传递输入尺寸
            pretrained=args.pretrained
        )
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练
        trainer = DomainGeneralizationTrainer(
            model, train_loaders, test_loaders, device
        )
        
        train_history, test_history = trainer.train(args.epochs)
        
        # 保存训练历史
        results = {
            'args': vars(args),
            'train_history': train_history,
            'test_history': test_history,
            'final_test_accuracy': test_history['accuracy'][-1]
        }
        
        with open('vgg16_domain_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("结果已保存到 vgg16_domain_results.json")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
