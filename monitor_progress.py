#!/usr/bin/env python3
"""
实验进度监控脚本
实时显示训练进度、显存使用和性能指标
"""

import time
import subprocess
import psutil
import json
from pathlib import Path
import torch

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                memory_used, memory_total, gpu_util = line.split(', ')
                gpu_info.append({
                    'memory_used': int(memory_used),
                    'memory_total': int(memory_total),
                    'memory_percent': int(memory_used) / int(memory_total) * 100,
                    'gpu_utilization': int(gpu_util)
                })
            return gpu_info
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
    return []

def get_latest_results():
    """获取最新的实验结果"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # 查找最新的结果目录
    comparison_dirs = list(results_dir.glob("comparison_*"))
    if not comparison_dirs:
        return None
    
    latest_dir = max(comparison_dirs, key=lambda x: x.stat().st_mtime)
    
    # 查看是否有结果文件
    results_file = latest_dir / "comparison_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return None

def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    """主监控循环"""
    print("🚀 实验进度监控器启动")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        try:
            # 清屏
            print("\033[2J\033[H", end="")
            
            # 显示标题
            elapsed = time.time() - start_time
            print(f"🚀 ResNet34 vs Self-Attention ResNet34 实验监控")
            print(f"⏱️  运行时间: {format_time(elapsed)}")
            print("=" * 60)
            
            # GPU信息
            gpu_info = get_gpu_info()
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    print(f"🎮 GPU {i}: {gpu['memory_used']:4d}MB/{gpu['memory_total']:4d}MB "
                          f"({gpu['memory_percent']:5.1f}%) | 利用率: {gpu['gpu_utilization']:3d}%")
            
            # 系统信息
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            print(f"💻 CPU: {cpu_percent:5.1f}% | RAM: {memory.used//1024//1024:5d}MB/{memory.total//1024//1024:5d}MB ({memory.percent:5.1f}%)")
            
            print("-" * 60)
            
            # 实验结果
            results = get_latest_results()
            if results:
                completed = len([r for r in results if r.get('success', False)])
                total_experiments = 6  # 2 models * 3 test_envs
                print(f"📊 实验进度: {completed}/{total_experiments} 完成")
                
                for result in results:
                    if result.get('success', False):
                        model = result['model_type']
                        test_env = result['test_env']
                        acc = result['test_accuracy']
                        time_taken = result['training_time']
                        print(f"✅ {model} (env {test_env}): {acc:.4f} ({format_time(time_taken)})")
                    elif 'error' in result:
                        model = result['model_type']
                        test_env = result['test_env']
                        print(f"❌ {model} (env {test_env}): 失败")
            else:
                print("📊 等待实验结果...")
            
            print("-" * 60)
            print("按 Ctrl+C 退出监控")
            
            time.sleep(5)  # 每5秒更新一次
            
        except KeyboardInterrupt:
            print("\n👋 监控器已停止")
            break
        except Exception as e:
            print(f"监控错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
