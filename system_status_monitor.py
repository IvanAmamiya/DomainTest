#!/usr/bin/env python3
"""
综合系统状态监控
监控实验进度、GPU状态、自动git推送等
"""

import os
import time
import json
import subprocess
from datetime import datetime
import psutil

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
            for line in lines:
                memory_used, memory_total, gpu_util, temp = line.split(', ')
                return {
                    'memory_used': memory_used,
                    'memory_total': memory_total, 
                    'gpu_utilization': gpu_util,
                    'temperature': temp
                }
        return None
    except:
        return None

def check_experiment_running():
    """检查实验是否在运行"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'comparison_experiment.py' in cmdline:
                    return {
                        'running': True,
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'cpu_percent': proc.cpu_percent(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024
                    }
        except:
            continue
    return {'running': False}

def check_git_push_service():
    """检查自动git推送服务状态"""
    pid_file = '/home/ribiki/  DomainTest/auto_git_push.pid'
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                return {
                    'running': True,
                    'pid': pid,
                    'cpu_percent': proc.cpu_percent(),
                    'memory_mb': proc.memory_info().rss / 1024 / 1024
                }
        except:
            pass
    return {'running': False}

def get_latest_results():
    """获取最新的实验结果"""
    results_dir = '/home/ribiki/  DomainTest/results'
    latest_dirs = []
    
    try:
        for item in os.listdir(results_dir):
            if item.startswith('comparison_20250530'):
                path = os.path.join(results_dir, item)
                if os.path.isdir(path):
                    latest_dirs.append((item, os.path.getctime(path)))
        
        latest_dirs.sort(key=lambda x: x[1], reverse=True)
        
        if latest_dirs:
            latest_dir = latest_dirs[0][0]
            return {
                'latest_experiment': latest_dir,
                'creation_time': datetime.fromtimestamp(latest_dirs[0][1]).strftime('%H:%M:%S'),
                'total_experiments_today': len(latest_dirs)
            }
    except:
        pass
    
    return {'latest_experiment': 'None', 'creation_time': 'N/A', 'total_experiments_today': 0}

def print_status():
    """打印系统状态"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("=" * 80)
    print(f"🖥️  系统状态监控 - {timestamp}")
    print("=" * 80)
    
    # GPU状态
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"🎮 GPU状态:")
        print(f"   内存使用: {gpu_info['memory_used']} / {gpu_info['memory_total']}")
        print(f"   GPU利用率: {gpu_info['gpu_utilization']}")
        print(f"   温度: {gpu_info['temperature']}")
    else:
        print("🎮 GPU状态: 无法获取")
    
    print()
    
    # 实验状态
    exp_info = check_experiment_running()
    if exp_info['running']:
        print(f"🧪 实验状态: ✅ 运行中")
        print(f"   PID: {exp_info['pid']}")
        print(f"   CPU使用: {exp_info['cpu_percent']:.1f}%")
        print(f"   内存使用: {exp_info['memory_mb']:.1f} MB")
    else:
        print("🧪 实验状态: ❌ 未运行")
    
    print()
    
    # Git推送服务状态
    git_info = check_git_push_service()
    if git_info['running']:
        print(f"📤 Git推送服务: ✅ 运行中")
        print(f"   PID: {git_info['pid']}")
        print(f"   CPU使用: {git_info['cpu_percent']:.1f}%")
        print(f"   内存使用: {git_info['memory_mb']:.1f} MB")
    else:
        print("📤 Git推送服务: ❌ 未运行")
    
    print()
    
    # 实验结果
    results_info = get_latest_results()
    print(f"📊 实验结果:")
    print(f"   最新实验: {results_info['latest_experiment']}")
    print(f"   创建时间: {results_info['creation_time']}")
    print(f"   今日实验总数: {results_info['total_experiments_today']}")
    
    print("=" * 80)

def main():
    """主监控循环"""
    print("启动系统状态监控...")
    print("按 Ctrl+C 停止监控")
    print()
    
    try:
        while True:
            print_status()
            print("⏰ 下次更新: 30秒后")
            print()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main()
