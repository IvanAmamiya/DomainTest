import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import iqr

# 路径配置
data_path = "results/comparison_20250531_014709/comparison_results.json"

with open(data_path, 'r') as f:
    results = json.load(f)

# 只分析ResNet18和Self-Attention ResNet18在test_env=0的表现
resnet18 = None
sa_resnet18 = None
for r in results:
    if r['model_type'] == 'resnet18' and r['test_env'] == 0:
        resnet18 = r
    if r['model_type'] == 'selfattentionresnet18' and r['test_env'] == 0:
        sa_resnet18 = r

# 画图
plt.figure(figsize=(12, 6))

# 训练损失
plt.subplot(2, 2, 1)
plt.plot(resnet18['train_history']['loss'], label='ResNet18 Train Loss', color='red')
plt.plot(sa_resnet18['train_history']['loss'], label='SA-ResNet18 Train Loss', color='blue')
plt.title('Train Loss')
plt.legend()

# 测试损失
plt.subplot(2, 2, 2)
plt.plot(resnet18['test_history']['loss'], label='ResNet18 Test Loss', color='orange')
plt.plot(sa_resnet18['test_history']['loss'], label='SA-ResNet18 Test Loss', color='green')
plt.title('Test Loss')
plt.legend()

# 训练精度
plt.subplot(2, 2, 3)
plt.plot(resnet18['train_history']['accuracy'], label='ResNet18 Train Acc', color='purple')
plt.plot(sa_resnet18['train_history']['accuracy'], label='SA-ResNet18 Train Acc', color='cyan')
plt.title('Train Accuracy')
plt.legend()

# 测试精度
plt.subplot(2, 2, 4)
plt.plot(resnet18['test_history']['accuracy'], label='ResNet18 Test Acc', color='brown')
plt.plot(sa_resnet18['test_history']['accuracy'], label='SA-ResNet18 Test Acc', color='lime')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/clear_comparison_plot.png', dpi=200)
plt.show()

print("图已保存为 results/comparison_20250531_014709/clear_comparison_plot.png")

# ================== 新增：不同Env对比 =====================
# 收集不同env下的测试精度
envs = [0, 1, 2]
resnet18_acc = []
sa_resnet18_acc = []
for env in envs:
    for r in results:
        if r['model_type'] == 'resnet18' and r['test_env'] == env:
            resnet18_acc.append(r['test_accuracy'])
        if r['model_type'] == 'selfattentionresnet18' and r['test_env'] == env:
            sa_resnet18_acc.append(r['test_accuracy'])

plt.figure(figsize=(6, 5))
plt.plot(envs, resnet18_acc, marker='o', color='red', label='ResNet18')
plt.plot(envs, sa_resnet18_acc, marker='s', color='blue', label='SA-ResNet18')
plt.xticks(envs)
plt.xlabel('Test Env')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy across Different Envs')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/env_comparison_plot.png', dpi=200)
plt.show()

print("Env对比图已保存为 results/comparison_20250531_014709/env_comparison_plot.png")

# ================== 新增：不同Env下测试集精度/loss稳定性分析 =====================
def remove_outliers_iqr(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr_val = q3 - q1
    lower = q1 - 1.5 * iqr_val
    upper = q3 + 1.5 * iqr_val
    return data[(data >= lower) & (data <= upper)]

# 收集所有env下每个epoch的test accuracy/loss
envs = [0, 1, 2]
models = ['resnet18', 'selfattentionresnet18']
acc_stats = {m: [] for m in models}
loss_stats = {m: [] for m in models}

for env in envs:
    for m in models:
        # 找到该模型该env的结果
        for r in results:
            if r['model_type'] == m and r['test_env'] == env:
                test_acc = r['test_history']['accuracy']
                test_loss = r['test_history']['loss']
                # 剔除离群值
                acc_wo = remove_outliers_iqr(test_acc)
                loss_wo = remove_outliers_iqr(test_loss)
                # 统计均值和std
                acc_stats[m].append((np.mean(acc_wo), np.std(acc_wo)))
                loss_stats[m].append((np.mean(loss_wo), np.std(loss_wo)))

# 绘制误差棒图
x = np.arange(len(envs))
width = 0.35
plt.figure(figsize=(10, 5))

# 精度
plt.subplot(1, 2, 1)
for i, m in enumerate(models):
    means = [acc_stats[m][j][0] for j in range(len(envs))]
    stds = [acc_stats[m][j][1] for j in range(len(envs))]
    plt.bar(x + i*width, means, width, yerr=stds, capsize=5, label=m)
plt.xticks(x + width/2, [f"Env {e}" for e in envs])
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Stability (IQR去离群)')
plt.legend()
plt.ylim(0, 1)

# Loss
plt.subplot(1, 2, 2)
for i, m in enumerate(models):
    means = [loss_stats[m][j][0] for j in range(len(envs))]
    stds = [loss_stats[m][j][1] for j in range(len(envs))]
    plt.bar(x + i*width, means, width, yerr=stds, capsize=5, label=m)
plt.xticks(x + width/2, [f"Env {e}" for e in envs])
plt.ylabel('Test Loss')
plt.title('Test Loss Stability (IQR去离群)')
plt.legend()

plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/env_stability_iqr.png', dpi=200)
plt.show()

print("不同env下测试集精度/loss稳定性误差棒图已保存为 results/comparison_20250531_014709/env_stability_iqr.png")

# ========== 追加env=1和env=2的训练/测试曲线 =============
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
for idx, env in enumerate([0, 1, 2]):
    resnet = None
    sa_resnet = None
    for r in results:
        if r['model_type'] == 'resnet18' and r['test_env'] == env:
            resnet = r
        if r['model_type'] == 'selfattentionresnet18' and r['test_env'] == env:
            sa_resnet = r
    if resnet is None or sa_resnet is None:
        continue
    # 训练/测试损失
    axes[idx, 0].plot(resnet['train_history']['loss'], label='ResNet18 Train Loss', color='red', alpha=0.7)
    axes[idx, 0].plot(resnet['test_history']['loss'], label='ResNet18 Test Loss', color='orange', alpha=0.7)
    axes[idx, 0].plot(sa_resnet['train_history']['loss'], label='SA-ResNet18 Train Loss', color='blue', alpha=0.7)
    axes[idx, 0].plot(sa_resnet['test_history']['loss'], label='SA-ResNet18 Test Loss', color='green', alpha=0.7)
    axes[idx, 0].set_title(f'Env {env} Loss')
    axes[idx, 0].legend(fontsize=8)
    axes[idx, 0].set_ylabel('Loss')
    axes[idx, 0].set_xlabel('Epoch')
    axes[idx, 0].grid(True, alpha=0.3)
    # 训练/测试精度
    axes[idx, 1].plot(resnet['train_history']['accuracy'], label='ResNet18 Train Acc', color='purple', alpha=0.7)
    axes[idx, 1].plot(resnet['test_history']['accuracy'], label='ResNet18 Test Acc', color='brown', alpha=0.7)
    axes[idx, 1].plot(sa_resnet['train_history']['accuracy'], label='SA-ResNet18 Train Acc', color='cyan', alpha=0.7)
    axes[idx, 1].plot(sa_resnet['test_history']['accuracy'], label='SA-ResNet18 Test Acc', color='lime', alpha=0.7)
    axes[idx, 1].set_title(f'Env {env} Accuracy')
    axes[idx, 1].legend(fontsize=8)
    axes[idx, 1].set_ylabel('Accuracy')
    axes[idx, 1].set_xlabel('Epoch')
    axes[idx, 1].set_ylim(0, 1)
    axes[idx, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/all_envs_train_test_curves.png', dpi=200)
plt.show()
print('所有env下训练/测试曲线已保存为 results/comparison_20250531_014709/all_envs_train_test_curves.png')

# ========== 新增：分开绘制训练和测试曲线，增强颜色对比 =============
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for idx, env in enumerate([0, 1, 2]):
    resnet = None
    sa_resnet = None
    for r in results:
        if r['model_type'] == 'resnet18' and r['test_env'] == env:
            resnet = r
        if r['model_type'] == 'selfattentionresnet18' and r['test_env'] == env:
            sa_resnet = r
    if resnet is None or sa_resnet is None:
        continue
    axes[idx].plot(resnet['train_history']['accuracy'], label='ResNet18 Train', color='#D62728', linewidth=2)
    axes[idx].plot(sa_resnet['train_history']['accuracy'], label='SA-ResNet18 Train', color='#1F77B4', linewidth=2)
    axes[idx].set_title(f'Env {env} Train Accuracy', fontsize=13)
    axes[idx].set_ylabel('Accuracy')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylim(0, 1)
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/all_envs_train_accuracy.png', dpi=200)
plt.show()
print('所有env下训练精度对比图已保存为 results/comparison_20250531_014709/all_envs_train_accuracy.png')

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for idx, env in enumerate([0, 1, 2]):
    resnet = None
    sa_resnet = None
    for r in results:
        if r['model_type'] == 'resnet18' and r['test_env'] == env:
            resnet = r
        if r['model_type'] == 'selfattentionresnet18' and r['test_env'] == env:
            sa_resnet = r
    if resnet is None or sa_resnet is None:
        continue
    axes[idx].plot(resnet['test_history']['accuracy'], label='ResNet18 Test', color='#FF7F0E', linewidth=2)
    axes[idx].plot(sa_resnet['test_history']['accuracy'], label='SA-ResNet18 Test', color='#2CA02C', linewidth=2)
    axes[idx].set_title(f'Env {env} Test Accuracy', fontsize=13)
    axes[idx].set_ylabel('Accuracy')
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylim(0, 1)
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/all_envs_test_accuracy.png', dpi=200)
plt.show()
print('所有env下测试精度对比图已保存为 results/comparison_20250531_014709/all_envs_test_accuracy.png')

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for idx, env in enumerate([0, 1, 2]):
    resnet = None
    sa_resnet = None
    for r in results:
        if r['model_type'] == 'resnet18' and r['test_env'] == env:
            resnet = r
        if r['model_type'] == 'selfattentionresnet18' and r['test_env'] == env:
            sa_resnet = r
    if resnet is None or sa_resnet is None:
        continue
    axes[idx].plot(resnet['train_history']['loss'], label='ResNet18 Train', color='#8C564B', linewidth=2)
    axes[idx].plot(sa_resnet['train_history']['loss'], label='SA-ResNet18 Train', color='#17BECF', linewidth=2)
    axes[idx].set_title(f'Env {env} Train Loss', fontsize=13)
    axes[idx].set_ylabel('Loss')
    axes[idx].set_xlabel('Epoch')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/all_envs_train_loss.png', dpi=200)
plt.show()
print('所有env下训练损失对比图已保存为 results/comparison_20250531_014709/all_envs_train_loss.png')

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
for idx, env in enumerate([0, 1, 2]):
    resnet = None
    sa_resnet = None
    for r in results:
        if r['model_type'] == 'resnet18' and r['test_env'] == env:
            resnet = r
        if r['model_type'] == 'selfattentionresnet18' and r['test_env'] == env:
            sa_resnet = r
    if resnet is None or sa_resnet is None:
        continue
    axes[idx].plot(resnet['test_history']['loss'], label='ResNet18 Test', color='#E377C2', linewidth=2)
    axes[idx].plot(sa_resnet['test_history']['loss'], label='SA-ResNet18 Test', color='#BCBD22', linewidth=2)
    axes[idx].set_title(f'Env {env} Test Loss', fontsize=13)
    axes[idx].set_ylabel('Loss')
    axes[idx].set_xlabel('Epoch')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('results/comparison_20250531_014709/all_envs_test_loss.png', dpi=200)
plt.show()
print('所有env下测试损失对比图已保存为 results/comparison_20250531_014709/all_envs_test_loss.png')
