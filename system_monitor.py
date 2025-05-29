#!/usr/bin/env python3
"""
综合监控脚本 - GPU使用率、实验进度和自动推送状态
"""

import subprocess
import time
import os
import json
import glob
from datetime import datetime

def run_command(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def get_gpu_info():
    """获取GPU信息"""
    success, stdout, stderr = run_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
    if success and stdout:
        lines = stdout.split('\n')
        gpu_info = []
        for i, line in enumerate(lines):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_info.append({
                        'gpu_id': i,
                        'utilization': f"{parts[0]}%",
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'memory_percent': round(int(parts[1]) / int(parts[2]) * 100, 1),
                        'temperature': f"{parts[3]}°C"
                    })
        return gpu_info
    return []

def get_running_experiments():
    """获取正在运行的实验"""
    success, stdout, stderr = run_command("ps aux | grep comparison_experiment | grep -v grep")
    if success and stdout:
        processes = []
        for line in stdout.split('\n'):
            if 'python' in line and 'comparison_experiment' in line:
                parts = line.split()
                if len(parts) >= 11:
                    processes.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9],
                        'command': ' '.join(parts[10:])
                    })
        return processes
    return []

def get_latest_results():
    """获取最新的实验结果"""
    try:
        # 查找最新的结果目录
        result_dirs = glob.glob("results/comparison_*")
        if not result_dirs:
            return None
        
        latest_dir = max(result_dirs, key=os.path.getctime)
        
        # 查找该目录下的子实验
        sub_dirs = glob.glob(f"{latest_dir}/*/")
        completed_experiments = []
        
        for sub_dir in sub_dirs:
            # 检查是否有完成的结果文件
            result_files = glob.glob(f"{sub_dir}/complete_results_*.json")
            if result_files:
                latest_result = max(result_files, key=os.path.getctime)
                try:
                    with open(latest_result, 'r') as f:
                        data = json.load(f)
                        completed_experiments.append({
                            'experiment': os.path.basename(sub_dir),
                            'file': latest_result,
                            'timestamp': data.get('timestamp', 'Unknown'),
                            'test_accuracy': data.get('test_accuracy', 'Unknown'),
                            'train_accuracy': data.get('train_accuracy', 'Unknown'),
                            'epochs_completed': data.get('epochs_completed', 'Unknown')
                        })
                except:
                    pass
        
        return {
            'experiment_dir': latest_dir,
            'completed': completed_experiments,
            'total_subdirs': len(sub_dirs)
        }
    except Exception as e:
        return None

def check_auto_git_push():
    """检查自动git推送服务状态"""
    if os.path.exists("auto_git_push.pid"):
        try:
            with open("auto_git_push.pid", 'r') as f:
                pid = f.read().strip()
            
            success, _, _ = run_command(f"ps -p {pid}")
            if success:
                return {'status': 'running', 'pid': pid}
            else:
                return {'status': 'stopped', 'pid': pid}
        except:
            return {'status': 'error', 'pid': None}
    else:
        return {'status': 'not_started', 'pid': None}

def main():
    """主监控函数"""
    print("="*80)
    print(f"🚀 DomainTest 实验监控面板")
    print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # GPU状态
    print("\n📊 GPU 状态:")
    gpu_info = get_gpu_info()
    if gpu_info:
        for gpu in gpu_info:
            print(f"  GPU {gpu['gpu_id']}: {gpu['utilization']} 使用率 | "
                  f"{gpu['memory_used']}/{gpu['memory_total']}MB ({gpu['memory_percent']}%) | "
                  f"{gpu['temperature']}")
    else:
        print("  ❌ 无法获取GPU信息")
    
    # 运行中的实验
    print("\n🧪 运行中的实验:")
    experiments = get_running_experiments()
    if experiments:
        print(f"  发现 {len(experiments)} 个运行中的实验进程:")
        for exp in experiments:
            print(f"    PID {exp['pid']}: CPU {exp['cpu']}% | MEM {exp['mem']}% | 运行时间 {exp['time']}")
    else:
        print("  ❌ 没有发现运行中的实验")
    
    # 最新结果
    print("\n📈 最新实验结果:")
    results = get_latest_results()
    if results:
        print(f"  实验目录: {results['experiment_dir']}")
        print(f"  完成的实验: {len(results['completed'])}/{results['total_subdirs']}")
        
        if results['completed']:
            print("  已完成的实验:")
            for exp in results['completed']:
                print(f"    • {exp['experiment']}: "
                      f"测试准确率 {exp['test_accuracy']}, "
                      f"训练准确率 {exp['train_accuracy']}, "
                      f"完成 {exp['epochs_completed']} 轮")
    else:
        print("  ❌ 没有找到实验结果")
    
    # 自动Git推送状态
    print("\n🔄 自动Git推送状态:")
    git_status = check_auto_git_push()
    if git_status['status'] == 'running':
        print(f"  ✅ 服务正在运行 (PID: {git_status['pid']})")
        print("  📝 每30分钟自动推送一次")
    elif git_status['status'] == 'stopped':
        print(f"  ⚠️  服务已停止 (PID: {git_status['pid']})")
    else:
        print("  ❌ 服务未启动")
    
    print("\n" + "="*80)
    print("🔧 控制命令:")
    print("  监控实验: watch -n 30 python3 system_monitor.py")
    print("  查看日志: tail -f auto_git_push.log")
    print("  停止推送: ./stop_auto_git_push.sh")
    print("  启动推送: ./start_auto_git_push.sh")
    print("="*80)

if __name__ == "__main__":
    main()
