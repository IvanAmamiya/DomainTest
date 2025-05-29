#!/usr/bin/env python3
"""
GPU和显存监控工具
"""

import torch
import time
import os
import subprocess
import psutil
from datetime import datetime


class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
    def get_gpu_memory_info(self):
        """获取GPU显存信息"""
        if not self.gpu_available:
            return {"error": "CUDA不可用"}
        
        # PyTorch显存信息
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        
        # 转换为MB
        allocated_mb = allocated / 1024**2
        reserved_mb = reserved / 1024**2
        total_mb = total_memory / 1024**2
        free_mb = total_mb - reserved_mb
        
        return {
            "device": torch.cuda.get_device_name(self.device),
            "allocated_mb": allocated_mb,
            "reserved_mb": reserved_mb,
            "total_mb": total_mb,
            "free_mb": free_mb,
            "utilization_percent": (reserved_mb / total_mb) * 100
        }
    
    def get_nvidia_smi_info(self):
        """使用nvidia-smi获取GPU信息"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            
            for i, line in enumerate(lines):
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 6:
                    gpu_info.append({
                        "gpu_id": i,
                        "name": parts[0],
                        "total_memory_mb": float(parts[1]),
                        "used_memory_mb": float(parts[2]),
                        "free_memory_mb": float(parts[3]),
                        "gpu_utilization_percent": float(parts[4]),
                        "temperature_c": float(parts[5])
                    })
            
            return gpu_info
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {"error": f"nvidia-smi命令失败: {e}"}
    
    def print_memory_summary(self):
        """打印显存使用摘要"""
        print(f"\n{'='*60}")
        print(f"GPU显存监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        if not self.gpu_available:
            print("❌ CUDA不可用，无法监控GPU")
            return
        
        # PyTorch显存信息
        pytorch_info = self.get_gpu_memory_info()
        print(f"🔹 设备: {pytorch_info['device']}")
        print(f"🔹 PyTorch显存:")
        print(f"   - 已分配: {pytorch_info['allocated_mb']:.1f} MB")
        print(f"   - 已保留: {pytorch_info['reserved_mb']:.1f} MB") 
        print(f"   - 总容量: {pytorch_info['total_mb']:.1f} MB")
        print(f"   - 使用率: {pytorch_info['utilization_percent']:.1f}%")
        
        # nvidia-smi信息
        nvidia_info = self.get_nvidia_smi_info()
        if isinstance(nvidia_info, list) and len(nvidia_info) > 0:
            gpu = nvidia_info[0]  # 假设使用第一个GPU
            print(f"\n🔹 nvidia-smi显存:")
            print(f"   - 已使用: {gpu['used_memory_mb']:.1f} MB")
            print(f"   - 可用: {gpu['free_memory_mb']:.1f} MB")
            print(f"   - 总容量: {gpu['total_memory_mb']:.1f} MB")
            print(f"   - GPU利用率: {gpu['gpu_utilization_percent']:.1f}%")
            print(f"   - 温度: {gpu['temperature_c']:.1f}°C")
        
        print(f"{'='*60}")
    
    def monitor_during_training(self, interval=10):
        """训练期间持续监控"""
        print(f"开始GPU监控，每{interval}秒更新一次...")
        print("按Ctrl+C停止监控")
        
        try:
            while True:
                self.print_memory_summary()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
    
    def clear_cache(self):
        """清理PyTorch缓存"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            print("✅ PyTorch GPU缓存已清理")
        else:
            print("❌ CUDA不可用，无需清理缓存")


def monitor_memory_usage(func):
    """装饰器：监控函数执行期间的显存使用"""
    def wrapper(*args, **kwargs):
        monitor = GPUMonitor()
        
        print(f"\n🚀 开始执行函数: {func.__name__}")
        monitor.print_memory_summary()
        
        result = func(*args, **kwargs)
        
        print(f"\n✅ 函数执行完成: {func.__name__}")
        monitor.print_memory_summary()
        
        return result
    return wrapper


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU显存监控工具")
    parser.add_argument("--monitor", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=int, default=5, help="监控间隔(秒)")
    parser.add_argument("--clear", action="store_true", help="清理PyTorch缓存")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor()
    
    if args.clear:
        monitor.clear_cache()
    
    if args.monitor:
        monitor.monitor_during_training(args.interval)
    else:
        monitor.print_memory_summary()
