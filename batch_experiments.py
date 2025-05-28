#!/usr/bin/env python3
"""
批量实验脚本 - 自动运行多个数据集和测试环境的组合
"""

import subprocess
import json
import time
import os
from datetime import datetime

# 实验配置
EXPERIMENTS = [
    # ColoredMNIST - 快速测试
    {
        'dataset': 'ColoredMNIST',
        'test_envs': [0, 1, 2],
        'epochs': 15,
        'batch_size': 128,
        'description': '彩色MNIST快速测试'
    },
    
    # TerraIncognita - 标准评估
    {
        'dataset': 'TerraIncognita', 
        'test_envs': [0, 1, 2, 3],
        'epochs': 30,
        'batch_size': 32,
        'description': '野生动物图像标准评估'
    }
]

def run_experiment(config, test_env):
    """运行单个实验"""
    dataset = config['dataset']
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/{dataset}_env{test_env}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"开始实验: {config['description']}")
    print(f"数据集: {dataset}, 测试环境: {test_env}")
    print(f"训练轮数: {epochs}, 批大小: {batch_size}")
    print(f"结果目录: {exp_dir}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        'python', 'vgg16_domain_test.py',
        '--dataset', dataset,
        '--test_env', str(test_env),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--pretrained'
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行实验
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=3600  # 1小时超时
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 移动结果文件到实验目录
        if os.path.exists('best_vgg16_domain_model.pth'):
            os.rename('best_vgg16_domain_model.pth', 
                     f'{exp_dir}/best_model.pth')
        
        if os.path.exists('vgg16_domain_results.json'):
            os.rename('vgg16_domain_results.json', 
                     f'{exp_dir}/results.json')
        
        # 保存实验日志
        log_data = {
            'config': config,
            'test_env': test_env,
            'duration_seconds': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        with open(f'{exp_dir}/experiment_log.json', 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        # 提取最终结果
        if result.returncode == 0:
            try:
                with open(f'{exp_dir}/results.json', 'r') as f:
                    results = json.load(f)
                    final_acc = results.get('final_test_accuracy', 0)
                    print(f"✅ 实验成功完成!")
                    print(f"   最终测试准确率: {final_acc:.4f}")
                    print(f"   训练时间: {duration/60:.1f} 分钟")
                    return final_acc
            except:
                print(f"⚠️  实验完成但无法读取结果")
                return 0
        else:
            print(f"❌ 实验失败 (返回码: {result.returncode})")
            print(f"错误信息: {result.stderr[:500]}...")
            return 0
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 实验超时 (1小时)")
        return 0
    except Exception as e:
        print(f"❌ 实验异常: {e}")
        return 0

def main():
    print("🚀 开始批量领域泛化实验")
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建实验根目录
    os.makedirs('experiments', exist_ok=True)
    
    # 运行所有实验
    all_results = []
    
    for config in EXPERIMENTS:
        dataset = config['dataset']
        test_envs = config['test_envs']
        
        print(f"\n🎯 开始数据集: {dataset}")
        print(f"测试环境: {test_envs}")
        
        dataset_results = []
        
        for test_env in test_envs:
            accuracy = run_experiment(config, test_env)
            dataset_results.append({
                'test_env': test_env,
                'accuracy': accuracy
            })
        
        # 计算平均性能
        avg_accuracy = sum(r['accuracy'] for r in dataset_results) / len(dataset_results)
        
        result_summary = {
            'dataset': dataset,
            'description': config['description'],
            'test_envs': test_envs,
            'individual_results': dataset_results,
            'average_accuracy': avg_accuracy
        }
        
        all_results.append(result_summary)
        
        print(f"\n📊 {dataset} 汇总结果:")
        for r in dataset_results:
            print(f"   环境 {r['test_env']}: {r['accuracy']:.4f}")
        print(f"   平均准确率: {avg_accuracy:.4f}")
    
    # 保存总结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"experiments/batch_results_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'total_experiments': sum(len(config['test_envs']) for config in EXPERIMENTS),
        'results': all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印最终总结
    print(f"\n{'='*80}")
    print("🎉 所有实验完成！")
    print(f"总结果已保存到: {summary_file}")
    print(f"{'='*80}")
    
    print("\n📈 最终结果汇总:")
    for result in all_results:
        print(f"\n{result['dataset']} ({result['description']}):")
        print(f"  平均准确率: {result['average_accuracy']:.4f}")
        for r in result['individual_results']:
            print(f"  环境 {r['test_env']}: {r['accuracy']:.4f}")

if __name__ == '__main__':
    main()
