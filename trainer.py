#!/usr/bin/env python3
"""
训练器模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time


class DomainGeneralizationTrainer:
    """领域泛化训练器"""
    
    def __init__(self, model, train_loaders, test_loaders, config, device='cuda'):
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.config = config
        self.device = device
        self.model.to(device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self._setup_optimizer()
        
        # 训练历史
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.epoch_times = []
        
    def _setup_optimizer(self):
        """设置优化器和调度器"""
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['training']['step_size'],
            gamma=self.config['training']['gamma']
        )
    
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
        
        for x, y in tqdm(all_batches, desc="Training", leave=False):
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
        
        avg_loss = total_loss / len(all_batches) if all_batches else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
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
                    'loss': env_loss / len(loader) if len(loader) > 0 else 0,
                    'samples': env_samples
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
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 评估测试集
            test_loss, test_acc, test_env_results = self.evaluate(
                self.test_loaders, "test"
            )
            
            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.test_history['loss'].append(test_loss)
            self.test_history['accuracy'].append(test_acc)
            
            # 记录各环境结果
            for i, result in enumerate(test_env_results):
                env_key = f'test_env_{result["env"]}_accuracy'
                if env_key not in self.test_history:
                    self.test_history[env_key] = []
                self.test_history[env_key].append(result['accuracy'])
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录epoch时间
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # 打印结果
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            print(f"学习率: {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"Epoch时间: {epoch_time:.2f}s")
            
            # 打印各环境的详细结果
            if len(test_env_results) > 1:
                print("各环境测试结果:")
                for result in test_env_results:
                    print(f"  环境 {result['env']}: Acc: {result['accuracy']:.4f} ({result['samples']} samples)")
            
            # 保存最好的模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self._save_best_model()
                print(f"★ 保存最佳模型 (测试准确率: {best_test_acc:.4f})")
        
        total_time = time.time() - start_time
        print(f"\n训练完成！")
        print(f"最佳测试准确率: {best_test_acc:.4f}")
        print(f"总训练时间: {total_time:.2f}s")
        print(f"平均每epoch时间: {np.mean(self.epoch_times):.2f}s")
        
        return self.train_history, self.test_history, best_test_acc
    
    def _save_best_model(self):
        """保存最佳模型"""
        model_path = self.config['output']['model_path']
        import os
        os.makedirs(model_path, exist_ok=True)
        
        model_file = os.path.join(model_path, 'best_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_info': self.model.get_model_info(),
            'config': self.config,
            'test_accuracy': max(self.test_history['accuracy']) if self.test_history['accuracy'] else 0
        }, model_file)
    
    def get_training_summary(self):
        """获取训练摘要"""
        if not self.train_history['accuracy']:
            return {}
        
        return {
            'final_train_acc': self.train_history['accuracy'][-1],
            'final_test_acc': self.test_history['accuracy'][-1],
            'best_test_acc': max(self.test_history['accuracy']),
            'final_train_loss': self.train_history['loss'][-1],
            'final_test_loss': self.test_history['loss'][-1],
            'total_epochs': len(self.train_history['accuracy']),
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'total_training_time': sum(self.epoch_times) if self.epoch_times else 0
        }
