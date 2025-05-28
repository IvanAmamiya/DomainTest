#!/usr/bin/env python3
"""
数据加载模块
"""

import sys
import os
from torch.utils.data import DataLoader

# 添加DomainBed路径
sys.path.append('./DomainBed')
from domainbed import datasets


class DomainDataLoader:
    """DomainBed数据集加载器"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.data_dir = config['dataset']['data_dir']
        self.test_envs = [config['dataset']['test_env']]
        
        # 构建DomainBed超参数
        self.hparams = {
            'batch_size': config['training']['batch_size'],
            'data_augmentation': True,
            'resnet18': False,
            'resnet_dropout': 0.0,
            'nonlinear_classifier': False,
            'class_balanced': False
        }
        
        # 加载数据集
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集"""
        try:
            self.dataset = datasets.get_dataset_class(self.dataset_name)(
                self.data_dir, self.test_envs, self.hparams
            )
            
            self.input_shape = self.dataset.input_shape
            self.num_classes = self.dataset.num_classes
            self.environment_names = self.dataset.ENVIRONMENTS
            
        except Exception as e:
            raise RuntimeError(f"加载数据集 {self.dataset_name} 失败: {e}")
    
    def get_dataloaders(self):
        """获取训练和测试的DataLoader"""
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['experiment']['num_workers']
        pin_memory = self.config['experiment']['pin_memory']
        
        train_loaders = []
        test_loaders = []
        
        for i, env_dataset in enumerate(self.dataset):
            loader = DataLoader(
                env_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            if i in self.test_envs:
                test_loaders.append(loader)
            else:
                train_loaders.append(loader)
        
        return train_loaders, test_loaders
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'name': self.dataset_name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'environment_names': self.environment_names,
            'test_envs': self.test_envs,
            'num_environments': len(self.dataset)
        }


def create_dataloader(config):
    """创建数据加载器"""
    return DomainDataLoader(config)
