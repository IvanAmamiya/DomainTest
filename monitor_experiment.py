#!/usr/bin/env python3
"""
实验监控脚本
定期检查实验进度并显示状态
"""

import time
import os
import json
from pathlib import Path
import subprocess
import sys

def check_experiment_status():
    """检查实验状态"""
    results_dir = Path("results")
    
    # 查找最新的实验目录
    comparison_dirs = list(results_dir.glob("comparison_*"))
    if comparison_dirs:
        latest_dir = max(comparison_dirs, key=lambda x: x.stat().st_mtime)
        print(f"最新实验目录: {latest_dir}")
        
        # 检查结果文件
        result_file = latest_dir / "comparison_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            print(f"已完成实验数量: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['model_type']} on {result['dataset']} (env {result['test_env']})")
                if result.get('success', False):
                    print(f"      测试准确率: {result['test_accuracy']:.4f}")
                    print(f"      训练时间: {result['training_time']:.2f}s")
                else:
                    print(f"      状态: 失败 - {result.get('error', '未知错误')}")
                print()
        else:
            print("暂无完成的实验结果")
    else:
        print("未找到实验目录")

def check_gpu_usage():
    """检查GPU使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                gpu_util, mem_used, mem_total = line.split(', ')
                print(f"GPU {i}: 使用率 {gpu_util}%, 内存 {mem_used}MB/{mem_total}MB")
        else:
            print("无法获取GPU信息")
    except Exception as e:
        print(f"GPU检查失败: {e}")

def monitor_experiment():
    """持续监控实验"""
    print("开始监控实验进度...")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"实验监控 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            check_experiment_status()
            print(f"\n{'-'*30}")
            check_gpu_usage()
            
            print(f"\n下次检查时间: {time.strftime('%H:%M:%S', time.localtime(time.time() + 300))}")
            time.sleep(300)  # 每5分钟检查一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # 只检查一次
        check_experiment_status()
        print()
        check_gpu_usage()
    else:
        # 持续监控
        monitor_experiment()
