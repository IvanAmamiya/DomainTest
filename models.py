#!/usr/bin/env python3
"""
VGG-16 模型定义模块
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGG16DomainModel(nn.Module):
    """基于VGG-16的领域泛化模型，适配不同尺寸图像"""
    
    def __init__(self, num_classes, input_channels=3, input_size=224, pretrained=True, dropout_rate=0.5):
        super(VGG16DomainModel, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # 如果输入尺寸很小（如MNIST的28x28），我们需要调整架构
        if input_size <= 32:
            self._build_small_vgg(input_channels, num_classes, dropout_rate)
        else:
            self._build_standard_vgg(input_channels, num_classes, pretrained, dropout_rate)
    
    def _build_small_vgg(self, input_channels, num_classes, dropout_rate):
        """构建适合小图像的VGG架构"""
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
    
    def _build_standard_vgg(self, input_channels, num_classes, pretrained, dropout_rate):
        """构建标准VGG-16架构"""
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
        """前向传播"""
        # 调整输入尺寸
        x = self._resize_input(x)
        
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def get_features(self, x):
        """提取特征向量，用于域对抗训练等高级方法"""
        x = self._resize_input(x)
        features = self.features(x)
        return features.view(features.size(0), -1)
    
    def _resize_input(self, x):
        """调整输入尺寸"""
        if self.input_size <= 32 and x.size(-1) != 28:
            x = torch.nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        elif self.input_size > 32 and x.size(-1) < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'architecture': 'VGG16_Small' if self.input_size <= 32 else 'VGG16_Standard'
        }


def create_model(config, num_classes, input_shape):
    """根据配置创建模型"""
    input_channels = config['model']['input_channels'] or input_shape[0]
    input_size = config['model']['input_size'] or max(input_shape[1], input_shape[2])
    
    model = VGG16DomainModel(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    return model
