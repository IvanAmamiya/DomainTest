#!/usr/bin/env python3
"""
模型定义模块
目前支持ResNet系列模型和Self-Attention ResNet模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


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
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重以提高数值稳定性"""
        for m in [self.query_conv, self.key_conv, self.value_conv, self.out_conv]:
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # 更小的gain
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 生成 Query, Key, Value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # 计算注意力权重 (QK^T)
        attention = torch.bmm(query, key)
        
        # 添加缩放以稳定Softmax - 使用 key 的维度
        key_dim = key.size(1)  # 这是 in_channels // reduction
        attention = attention / math.sqrt(key_dim)
        
        # 添加数值稳定性改进
        # 1. 梯度裁剪
        attention = torch.clamp(attention, min=-10, max=10)
        
        # 2. 检查NaN并处理
        if torch.isnan(attention).any():
            attention = torch.zeros_like(attention)
            
        attention = self.softmax(attention)
        
        # 3. 再次检查softmax后的NaN
        if torch.isnan(attention).any():
            attention = torch.ones_like(attention) / attention.size(-1)
        
        # 应用注意力权重
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # 输出投影和残差连接
        out = self.out_conv(out)
        
        # 4. 控制gamma的范围防止梯度爆炸
        gamma_clamped = torch.clamp(self.gamma, min=-1, max=1)
        out = gamma_clamped * out + x
        
        # 5. 最终检查输出
        if torch.isnan(out).any():
            return x  # 如果输出有NaN，直接返回输入（跳过attention）
        
        return out


class SelfAttentionResNet34(nn.Module):
    """带有Self-Attention机制的ResNet34"""
    def __init__(self, num_classes=10, input_channels=3, pretrained=True):
        super(SelfAttentionResNet34, self).__init__()
        
        # 加载预训练的ResNet34
        self.backbone = models.resnet34(pretrained=pretrained)
        
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


class SelfAttentionResNet50(nn.Module):
    """带有Self-Attention机制的ResNet50"""
    def __init__(self, num_classes=10, input_channels=3, pretrained=True):
        super(SelfAttentionResNet50, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
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
        self.attention1 = SelfAttentionModule(256)   # layer1 后
        self.attention2 = SelfAttentionModule(512)   # layer2 后  
        self.attention3 = SelfAttentionModule(1024)  # layer3 后
        self.attention4 = SelfAttentionModule(2048)  # layer4 后
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.attention1(x)
        x = self.backbone.layer2(x)
        x = self.attention2(x)
        x = self.backbone.layer3(x)
        x = self.attention3(x)
        x = self.backbone.layer4(x)
        x = self.attention4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x


class SelfAttentionResNet152(nn.Module):
    """带有Self-Attention机制的ResNet152"""
    def __init__(self, num_classes=10, input_channels=3, pretrained=True):
        super(SelfAttentionResNet152, self).__init__()
        self.backbone = models.resnet152(pretrained=pretrained)
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
        # 单头自注意力模块
        self.attention1 = SelfAttentionModule(64)
        self.attention2 = SelfAttentionModule(128)
        self.attention3 = SelfAttentionModule(256)
        self.attention4 = SelfAttentionModule(512)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.attention1(x)
        x = self.backbone.layer2(x)
        x = self.attention2(x)
        x = self.backbone.layer3(x)
        x = self.attention3(x)
        x = self.backbone.layer4(x)
        x = self.attention4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    """多头自注意力模块"""
    def __init__(self, in_channels, num_heads=4, reduction=8):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.reduction = reduction
        assert in_channels % num_heads == 0, "in_channels 必须能被 num_heads 整除"
        self.head_dim = in_channels // num_heads
        self.attentions = nn.ModuleList([
            SelfAttentionModule(self.head_dim, reduction) for _ in range(num_heads)
        ])
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 按通道分组
        x_split = torch.chunk(x, self.num_heads, dim=1)
        out_heads = [att(xi) for att, xi in zip(self.attentions, x_split)]
        out = torch.cat(out_heads, dim=1)
        out = self.out_proj(out)
        return out


class MultiHeadSelfAttentionResNet18(nn.Module):
    """带有多头自注意力机制的ResNet18"""
    def __init__(self, num_classes=10, input_channels=3, pretrained=True, num_heads=4):
        super(MultiHeadSelfAttentionResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
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
        self.attention1 = MultiHeadSelfAttentionModule(64, num_heads=num_heads)
        self.attention2 = MultiHeadSelfAttentionModule(128, num_heads=num_heads)
        self.attention3 = MultiHeadSelfAttentionModule(256, num_heads=num_heads)
        self.attention4 = MultiHeadSelfAttentionModule(512, num_heads=num_heads)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.attention1(x)
        x = self.backbone.layer2(x)
        x = self.attention2(x)
        x = self.backbone.layer3(x)
        x = self.attention3(x)
        x = self.backbone.layer4(x)
        x = self.attention4(x)
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
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
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


def create_self_attention_resnet34(num_classes, input_channels=3, pretrained=True):
    """创建Self-Attention ResNet34模型"""
    return SelfAttentionResNet34(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained
    )


def create_self_attention_resnet18(num_classes, input_channels=3, pretrained=True):
    """创建Self-Attention ResNet18模型"""
    return SelfAttentionResNet18(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained
    )


def create_self_attention_resnet50(num_classes, input_channels=3, pretrained=True):
    """创建Self-Attention ResNet50模型"""
    return SelfAttentionResNet50(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained
    )


def create_self_attention_resnet152(num_classes, input_channels=3, pretrained=True):
    """创建Self-Attention ResNet152模型"""
    return SelfAttentionResNet152(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained
    )


def create_multihead_self_attention_resnet18(num_classes, input_channels=3, pretrained=True, num_heads=4):
    """创建多头自注意力ResNet18模型"""
    return MultiHeadSelfAttentionResNet18(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained,
        num_heads=num_heads
    )


def create_model(config, num_classes, input_shape):
    """根据配置创建模型"""
    input_channels = config['model'].get('input_channels') or input_shape[0]
    model_type = config['model'].get('type', 'resnet34')
    pretrained = config['model'].get('pretrained', True)
    
    if model_type.startswith('resnet'):
        model = create_resnet_model(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            model_type=model_type
        )
    elif model_type == 'selfattentionresnet34':
        model = create_self_attention_resnet34(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained
        )
    elif model_type == 'selfattentionresnet18':
        model = create_self_attention_resnet18(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained
        )
    elif model_type == 'selfattentionresnet50':
        model = create_self_attention_resnet50(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained
        )
    elif model_type == 'multiheadselfattentionresnet18':
        num_heads = config['model'].get('num_heads', 4)
        model = create_multihead_self_attention_resnet18(
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            num_heads=num_heads
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def get_model_info(model, model_type='resnet34'):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 根据模型类型设置架构名称
    if model_type == 'selfattentionresnet34':
        architecture = 'Self-Attention ResNet34'
    elif model_type == 'selfattentionresnet18':
        architecture = 'Self-Attention ResNet18'
    elif model_type == 'selfattentionresnet50':
        architecture = 'Self-Attention ResNet50'
    elif model_type.startswith('resnet'):
        architecture = model_type.upper()
    else:
        architecture = model_type.upper()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'architecture': architecture
    }
