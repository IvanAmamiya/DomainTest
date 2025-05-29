# DomainTest: ResNet 域适应实验框架

一个用于在 DomainBed 数据集上测试 ResNet 系列模型的实验框架。

## 快速开始

```bash
git clone https://github.com/IvanAmamiya/DomainTest.git
cd DomainTest
./setup.sh  # 安装依赖和 DomainBed
python main.py  # 运行默认实验
```

## 支持的数据集

支持 DomainBed 的所有数据集：
- **ColoredMNIST**: 彩色 MNIST 数字
- **TerraIncognita**: 野生动物相机陷阱图像
- **PACS**: 照片、艺术画、卡通、素描
- **OfficeHome**: 办公室和家庭物体
- **VLCS**: PASCAL VOC, LabelMe, Caltech, SUN

## 使用方法

### 单个实验
```bash
python main.py --dataset ColoredMNIST --test_env 0
python main.py --dataset TerraIncognita --test_env 1
```

### 批量实验
```bash
python batch_experiments_v2.py  # 测试多种配置
```

### 查看结果
```bash
python main.py --summary     # 显示结果摘要
python main.py --comparison  # 生成对比图表
```

## 实验结果

结果自动保存至：
- `results/experiment_results.csv` - 结果汇总表
- `results/plots/` - 训练曲线和对比图
- `results/complete_results_*.json` - 详细实验数据

## 配置

编辑 `config.yaml` 或使用命令行参数：

```bash
python main.py --dataset ColoredMNIST --test_env 0 --epochs 5 --batch_size 32
```

## 致谢

本项目使用了 [DomainBed](https://github.com/facebookresearch/DomainBed) 的数据集。该框架实现了独立的 ResNet 训练系统和实验管理工具。
