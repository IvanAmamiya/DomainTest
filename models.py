#!/usr/bin/env python3
"""
模型定义模块
目前支持ResNet系列模型和Self-Attention ResNet模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SelfAttentionModule(nn.Module):
    """Self-Attention模块"""
    def __init__(self, in_channels, reduction=8):
        super(SelfAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Query, Key, Value 投影
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # 输出投影
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 生成 Query, Key, Value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # 计算注意力权重
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # 应用注意力权重
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # 输出投影和残差连接
        out = self.out_conv(out)
        out = self.gamma * out + x
        
        return out


class SelfAttentionResNet18(nn.Module):
    """带有Self-Attention机制的ResNet18"""
    def __init__(self, num_classes=10, input_channels=3, pretrained=True):
        super(SelfAttentionResNet18, self).__init__()
        
        # 加载预训练的ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 调整输入通道数
        if input_channels != 3:
            original_conv = self.backbone.conv1
            new_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            if pretrained and input_channels == 1:
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(torch.mean(original_conv.weight, dim=1, keepdim=True))
            elif pretrained and input_channels == 2:
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(original_conv.weight[:, :2, :, :])
            
            self.backbone.conv1 = new_conv
        
        # 添加Self-Attention模块到不同层
        self.attention1 = SelfAttentionModule(64)   # layer1 后
        self.attention2 = SelfAttentionModule(128)  # layer2 后  
        self.attention3 = SelfAttentionModule(256)  # layer3 后
        self.attention4 = SelfAttentionModule(512)  # layer4 后
        
        # 替换分类器
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        # 前向传播
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Layer 1 + Attention
        x = self.backbone.layer1(x)
        x = self.attention1(x)
        
        # Layer 2 + Attention
        x = self.backbone.layer2(x)
        x = self.attention2(x)
        
        # Layer 3 + Attention
        x = self.backbone.layer3(x)
        x = self.attention3(x)
        
        # Layer 4 + Attention
        x = self.backbone.layer4(x)
        x = self.attention4(x)
        
        # 全局平均池化和分类器
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        
        return x


def create_resnet_model(num_classes, input_channels=3, pretrained=True, model_type='resnet18'):
    """创建ResNet模型"""
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 如果输入通道数不是3，需要修改第一层
    if input_channels != 3:
        original_conv = model.conv1
        new_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained and input_channels == 1:
            # 对于单通道，使用预训练权重的平均值
            with torch.no_grad():
                new_conv.weight = nn.Parameter(torch.mean(original_conv.weight, dim=1, keepdim=True))
        elif pretrained and input_channels == 2:
            # 对于双通道，使用前两个通道
            with torch.no_grad():
                new_conv.weight = nn.Parameter(original_conv.weight[:, :2, :, :])
        
        model.conv1 = new_conv
    
    # 修改分类器
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def create_self_attention_resnet18(num_classes, input_channels=3, pretrained=True):
    """创建Self-Attention ResNet18模型"""
    return SelfAttentionResNet18(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained
    )


def create_model(config, num_classes, input_shape):
    """根据配置创建模型"""
    input_channels = config['model'].get('input_channels') or input_shape[0]
    model_type = config['model'].get('type', 'resnet18')
    pretrained = config['model'].get('pretrained', True)
    
    if model_type.startswith('resnet'):
        model = create_resnet_model(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            model_type=model_type
        )
    elif model_type == 'selfattentionresnet18':
        model = create_self_attention_resnet18(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def get_model_info(model, model_type='resnet18'):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 根据模型类型设置架构名称
    if model_type == 'selfattentionresnet18':
        architecture = 'Self-Attention ResNet18'
    elif model_type.startswith('resnet'):
        architecture = model_type.upper()
    else:
        architecture = model_type.upper()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'architecture': architecture
    }
