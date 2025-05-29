#!/usr/bin/env python3
"""
官方torchvision ResNet18的包装器，适配ColoredMNIST数据集
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TorchvisionResNet18Wrapper(nn.Module):
    """
    官方torchvision ResNet18的包装器，适配ColoredMNIST数据集
    解决输入通道和图像尺寸问题
    """
    
    def __init__(self, num_classes=2, input_channels=2, input_size=28, pretrained=True, dropout_rate=0.0):
        super(TorchvisionResNet18Wrapper, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_size = input_size
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # 创建ResNet18模型
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 修改第一层卷积以适应不同的输入通道数
        if input_channels != 3:
            # 为ColoredMNIST (28x28, 2通道) 优化的conv1
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            
            # 如果使用预训练权重，需要适配通道数
            if pretrained and input_channels == 2:
                # 取RGB权重的前两个通道的平均值作为初始权重
                with torch.no_grad():
                    original_weight = models.resnet18(pretrained=True).conv1.weight
                    # 对于2通道输入，使用前两个通道
                    new_weight = original_weight[:, :2, :, :]
                    self.resnet.conv1.weight.copy_(new_weight)
        
        # 去掉最大池化层以保持特征图尺寸（针对小图像）
        if input_size <= 32:
            self.resnet.maxpool = nn.Identity()
        
        # 修改最后的全连接层
        num_features = self.resnet.fc.in_features
        if dropout_rate > 0:
            self.resnet.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # 对于小图像，不需要上采样到224x224
        return self.resnet(x)
    
    def get_model_info(self):
        """返回模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'TorchvisionResNet18Wrapper',
            'architecture': 'ResNet18 (Official Torchvision)',
            'pretrained': self.pretrained,
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_groups': {
                'backbone': sum(p.numel() for name, p in self.named_parameters() if 'fc' not in name),
                'classifier': sum(p.numel() for name, p in self.named_parameters() if 'fc' in name)
            }
        }
