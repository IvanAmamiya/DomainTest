#!/usr/bin/env python3
"""
实时监控实验进展
显示当前训练状态、准确率、损失等信息
"""

import os
import time
import re
import subprocess
from datetime import datetime, timedelta

class RealTimeMonitor:
    def __init__(self, log_file_path=None):
        if log_file_path is None:
            # 自动查找最新的实验日志
            self.log_file_path = self.find_latest_log()
        else:
            self.log_file_path = log_file_path
        
        self.last_position = 0
        self.experiment_start_time = None
        self.current_epoch = 0
        self.total_epochs = 300
        self.latest_metrics = {}
        
    def find_latest_log(self):
        """查找最新的实验日志文件"""
        log_files = []
        for file in os.listdir('/home/ribiki/  DomainTest/'):
            if file.startswith('experiment_output_') and file.endswith('.log'):
                full_path = f'/home/ribiki/  DomainTest/{file}'
                log_files.append((full_path, os.path.getmtime(full_path)))
        
        if log_files:
            # 按修改时间排序，返回最新的
            log_files.sort(key=lambda x: x[1], reverse=True)
            return log_files[0][0]
        else:
            return None
    
    def clear_screen(self):
        """清屏"""
        os.system('clear')
    
    def get_gpu_status(self):
        """获取GPU使用状态"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:  # 跳过标题行
                    gpu_info = lines[1].split(', ')
                    if len(gpu_info) >= 4:
                        gpu_util = gpu_info[0].replace(' %', '')
                        mem_used = gpu_info[1].replace(' MiB', '')
                        mem_total = gpu_info[2].replace(' MiB', '')
                        temp = gpu_info[3].replace(' C', '')
                        return {
                            'utilization': gpu_util,
                            'memory_used': mem_used,
                            'memory_total': mem_total,
                            'temperature': temp
                        }
        except:
            pass
        return None
    
    def get_process_info(self):
        """获取实验进程信息"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'comparison_experiment.py' in line and 'python' in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        pid = parts[1]
                        cpu = parts[2]
                        mem = parts[3]
                        time_str = parts[9]
                        return {
                            'pid': pid,
                            'cpu': cpu,
                            'memory': mem,
                            'time': time_str
                        }
        except:
            pass
        return None
    
    def parse_log_line(self, line):
        """解析日志行，提取有用信息"""
        # 解析Epoch信息
        epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
            self.total_epochs = int(epoch_match.group(2))
        
        # 解析训练指标
        train_match = re.search(r'训练 - Loss: ([\d.]+), Acc: ([\d.]+)', line)
        if train_match:
            self.latest_metrics['train_loss'] = float(train_match.group(1))
            self.latest_metrics['train_acc'] = float(train_match.group(2))
        
        # 解析测试指标
        test_match = re.search(r'测试 - Loss: ([\d.]+), Acc: ([\d.]+)', line)
        if test_match:
            self.latest_metrics['test_loss'] = float(test_match.group(1))
            self.latest_metrics['test_acc'] = float(test_match.group(2))
        
        # 解析学习率
        lr_match = re.search(r'学习率: ([\d.]+)', line)
        if lr_match:
            self.latest_metrics['learning_rate'] = float(lr_match.group(1))
        
        # 解析Epoch时间
        time_match = re.search(r'Epoch时间: ([\d.]+)s', line)
        if time_match:
            self.latest_metrics['epoch_time'] = float(time_match.group(1))
    
    def read_new_lines(self):
        """读取日志文件的新行"""
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            return []
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
                return new_lines
        except:
            return []
    
    def calculate_eta(self):
        """计算预估完成时间"""
        if self.current_epoch > 0 and 'epoch_time' in self.latest_metrics:
            remaining_epochs = self.total_epochs - self.current_epoch
            remaining_time = remaining_epochs * self.latest_metrics['epoch_time']
            eta = datetime.now() + timedelta(seconds=remaining_time)
            return eta, remaining_time
        return None, None
    
    def display_status(self):
        """显示当前状态"""
        self.clear_screen()
        
        print("=" * 80)
        print("🚀 ResNet18 vs Self-Attention ResNet18 实验实时监控")
        print("=" * 80)
        
        # 显示当前时间
        print(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示日志文件
        if self.log_file_path:
            print(f"📋 监控日志: {os.path.basename(self.log_file_path)}")
        else:
            print("❌ 未找到实验日志文件")
            return
        
        print()
        
        # 显示进度信息
        progress = (self.current_epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0
        progress_bar = "█" * int(progress // 2) + "░" * (50 - int(progress // 2))
        
        print("📊 训练进度:")
        print(f"   Epoch: {self.current_epoch}/{self.total_epochs}")
        print(f"   进度: [{progress_bar}] {progress:.1f}%")
        
        # 计算ETA
        eta, remaining_time = self.calculate_eta()
        if eta and remaining_time:
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            print(f"   预估完成时间: {eta.strftime('%H:%M:%S')}")
            print(f"   剩余时间: {hours}h {minutes}m")
        
        print()
        
        # 显示最新指标
        if self.latest_metrics:
            print("📈 最新训练指标:")
            if 'train_loss' in self.latest_metrics and 'train_acc' in self.latest_metrics:
                print(f"   训练 - 损失: {self.latest_metrics['train_loss']:.4f}, 准确率: {self.latest_metrics['train_acc']:.4f}")
            
            if 'test_loss' in self.latest_metrics and 'test_acc' in self.latest_metrics:
                print(f"   测试 - 损失: {self.latest_metrics['test_loss']:.4f}, 准确率: {self.latest_metrics['test_acc']:.4f}")
            
            if 'learning_rate' in self.latest_metrics:
                print(f"   学习率: {self.latest_metrics['learning_rate']:.6f}")
            
            if 'epoch_time' in self.latest_metrics:
                print(f"   单轮时间: {self.latest_metrics['epoch_time']:.2f}秒")
        
        print()
        
        # 显示系统状态
        print("💻 系统状态:")
        
        # 进程信息
        process_info = self.get_process_info()
        if process_info:
            print(f"   进程ID: {process_info['pid']}")
            print(f"   CPU使用率: {process_info['cpu']}%")
            print(f"   内存使用率: {process_info['memory']}%")
            print(f"   运行时间: {process_info['time']}")
        else:
            print("   ❌ 未找到实验进程")
        
        # GPU信息
        gpu_info = self.get_gpu_status()
        if gpu_info:
            mem_percent = (int(gpu_info['memory_used']) / int(gpu_info['memory_total'])) * 100
            print(f"   GPU使用率: {gpu_info['utilization']}%")
            print(f"   GPU内存: {gpu_info['memory_used']}/{gpu_info['memory_total']} MiB ({mem_percent:.1f}%)")
            print(f"   GPU温度: {gpu_info['temperature']}°C")
        else:
            print("   GPU: 信息获取失败")
        
        print()
        print("⌨️  按 Ctrl+C 退出监控")
        print("=" * 80)
    
    def run(self, refresh_interval=2):
        """运行实时监控"""
        print("启动实验实时监控...")
        
        if not self.log_file_path:
            print("错误: 未找到实验日志文件")
            return
        
        try:
            while True:
                # 读取新的日志行
                new_lines = self.read_new_lines()
                for line in new_lines:
                    self.parse_log_line(line.strip())
                
                # 更新显示
                self.display_status()
                
                # 等待下次更新
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n监控已停止")
        except Exception as e:
            print(f"\n\n监控出错: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实时监控实验进展')
    parser.add_argument('--log', '-l', help='指定日志文件路径')
    parser.add_argument('--interval', '-i', type=int, default=2, help='刷新间隔(秒)')
    
    args = parser.parse_args()
    
    monitor = RealTimeMonitor(args.log)
    monitor.run(args.interval)

if __name__ == "__main__":
    main()
