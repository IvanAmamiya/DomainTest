# DomainTest: Comprehensive Domain Adaptation Experiment Framework

A modular and comprehensive experiment framework for testing domain adaptation algorithms on DomainBed datasets with VGG-16 architecture. This framework provides complete tools for single experiments, batch experiments, result logging, visualization, and performance comparison across different datasets and parameter combinations.

## 🌟 Key Features

- 🎯 **Modular Architecture**: Clean separation of data loading, model creation, training, and utilities
- 🏗️ **VGG-16 Implementation**: Adaptive VGG-16 architecture that handles both small (28x28) and large (224x224) images
- 📊 **Multi-Domain Support**: Train on multiple source domains and evaluate on specified test domains
- 🔄 **Batch Experiments**: Automated parameter grid search with comprehensive logging
- 📈 **Rich Visualization**: Training curves, comparison plots, and detailed performance analysis
- 💾 **Complete Logging**: CSV export, JSON results, and timestamp tracking
- ⚙️ **Flexible Configuration**: YAML-based configuration system with command-line overrides

## 📊 Supported Datasets

- **ColoredMNIST**: Colored MNIST digits (3 environments) - 28x28 images
- **TerraIncognita**: Wildlife camera trap images (4 environments) - 224x224 images  
- **PACS**: Photo, Art painting, Cartoon, Sketch (4 environments)
- **OfficeHome**: Office and home objects (4 environments)
- **VLCS**: PASCAL VOC, LabelMe, Caltech, SUN (4 environments)

## 🚀 Installation

### Option 1: Automatic Setup (Recommended)
```bash
git clone https://github.com/IvanAmamiya/DomainTest.git
cd DomainTest
./setup.sh
```

### Option 2: Manual Setup
```bash
git clone https://github.com/IvanAmamiya/DomainTest.git
cd DomainTest

# Install DomainTest requirements
pip install -r requirements.txt

# Clone and setup DomainBed
git clone https://github.com/facebookresearch/DomainBed.git
pip install -r DomainBed/requirements.txt
```

**Note:** The setup script will automatically:
- Clone the DomainBed repository (if not present)
- Install all required dependencies
- Verify the installation

## 🏃‍♂️ Quick Start

### Method 1: Using the Main Framework (Recommended)

```bash
# Single experiment with default config
python main.py

# Custom dataset and environment
python main.py --config config.yaml --dataset ColoredMNIST --test_env 1

# View experiment results summary
python main.py --summary

# Generate comparison plots
python main.py --comparison
```

### Method 2: Batch Experiments

```bash
# Run batch experiments with parameter grid search
python batch_experiments_v2.py

# The framework will automatically test combinations of:
# - Datasets: ColoredMNIST, TerraIncognita
# - Test environments: 0, 1
# - Learning rates: 0.0001, 0.0005
# - Batch sizes: 16, 32
```

### Method 3: Legacy Single Experiment

```bash
# Simple ColoredMNIST experiment
python vgg16_domain_test.py --dataset ColoredMNIST --test_env 0 --epochs 2

# TerraIncognita experiment with more epochs
python vgg16_domain_test.py --dataset TerraIncognita --test_env 0 --epochs 10
```

## 📈 Results and Analysis

The framework automatically generates comprehensive results:

### CSV Results (`results/experiment_results.csv`)
- Timestamp, experiment name, dataset info
- Model architecture and parameter counts  
- Training hyperparameters
- Final and best test accuracies
- Training time statistics

### Visualizations (`results/plots/`)
- Training curves for each experiment
- Comparison plots across different configurations
- Performance analysis by dataset and environment

### Example Results Summary
Based on our experiments:
- **ColoredMNIST Environment 0**: Best accuracy 72.67% (2 epochs, lr=0.0001, bs=16)
- **ColoredMNIST Environment 1**: Best accuracy ~50.39% (more challenging due to correlation shift)
- **TerraIncognita Environment 0**: Best accuracy 44.07% (1 epoch, baseline experiment)

## ⚙️ Configuration

### YAML Configuration (`config.yaml`)
```yaml
dataset:
  name: "ColoredMNIST"
  test_env: 0

model:
  architecture: "vgg16"
  pretrained: true
  dropout_rate: 0.5

training:
  epochs: 2
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 0.0001

# See config.yaml for full configuration options
```

### Command Line Arguments
```bash
python main.py --help  # See all available options
```

## 🏗️ Framework Architecture

```
DomainTest/
├── main.py                    # Main experiment runner
├── config_manager.py          # Configuration management
├── data_loader.py            # Dataset loading and preprocessing  
├── models.py                 # VGG-16 model implementation
├── trainer.py                # Training and validation logic
├── results_logger.py         # Result logging and visualization
├── batch_experiments_v2.py   # Batch experiment automation
├── config.yaml              # Default configuration
└── results/                  # Experiment outputs
    ├── experiment_results.csv
    ├── detailed_results.csv
    ├── complete_results_*.json
    └── plots/
```
- `--epochs`: 训练轮数
- `--batch_size`: 批大小
- `--device`: 设备 (cuda/cpu/auto)
- `--pretrained`: 使用预训练权重

## 输出文件

- `best_vgg16_domain_model.pth`: 最佳模型权重
- `vgg16_domain_results.json`: 训练历史和结果

## 实验建议

### ColoredMNIST (快速测试)
```bash
python vgg16_domain_test.py --dataset ColoredMNIST --test_env 0 --epochs 20 --batch_size 128
python vgg16_domain_test.py --dataset ColoredMNIST --test_env 1 --epochs 20 --batch_size 128
python vgg16_domain_test.py --dataset ColoredMNIST --test_env 2 --epochs 20 --batch_size 128
```

### TerraIncognita (标准评估)
```bash
python vgg16_domain_test.py --dataset TerraIncognita --test_env 0 --epochs 50 --batch_size 32 --pretrained
python vgg16_domain_test.py --dataset TerraIncognita --test_env 1 --epochs 50 --batch_size 32 --pretrained
python vgg16_domain_test.py --dataset TerraIncognita --test_env 2 --epochs 50 --batch_size 32 --pretrained
python vgg16_domain_test.py --dataset TerraIncognita --test_env 3 --epochs 50 --batch_size 32 --pretrained
```

## 框架架构

```
vgg16_domain_test.py
├── VGG16DomainModel        # VGG-16模型类
├── DomainDataLoader        # 数据加载器
├── DomainGeneralizationTrainer  # 训练器
└── main()                  # 主函数
```

## 扩展方向

1. **高级算法**: 可以在此基础上实现域对抗训练、IRM等算法
2. **其他模型**: 可以替换为ResNet、ViT等其他架构
3. **集成方法**: 可以实现模型集成和元学习方法
4. **评估指标**: 可以添加更多评估指标和可视化

## 性能基准

| 数据集 | 测试环境 | VGG-16准确率 | 备注 |
|--------|----------|--------------|------|
| ColoredMNIST | 0 | ~85% | 快速测试 |
| TerraIncognita | 0 | ~45% | 标准评估 |

## 故障排除

1. **CUDA内存不足**: 减小batch_size或使用CPU
2. **数据集未找到**: 检查data_dir路径
3. **导入错误**: 确认DomainBed在当前目录下

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures (ResNet, ViT, etc.)
- More domain adaptation algorithms (DANN, CORAL, etc.)
- Additional datasets and evaluation metrics
- Hyperparameter optimization integration
- Cross-dataset generalization studies

## 📄 License

This project is licensed under the MIT License. 

**Dependency License:** Please note that this project depends on [DomainBed](https://github.com/facebookresearch/DomainBed) for datasets, which has its own license terms.

## 📚 Citation

If you use this framework in your research, please consider citing:

```bibtex
@software{domaintest2025,
  title={DomainTest: Comprehensive Domain Adaptation Experiment Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/IvanAmamiya/DomainTest}
}
```

And also cite the original DomainBed paper:
```bibtex
@inproceedings{gulrajani2021search,
  title={In search of lost domain generalization},
  author={Gulrajani, Ishaan and Lopez-Paz, David},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## 🙏 Acknowledgments

- [DomainBed](https://github.com/facebookresearch/DomainBed) for providing the datasets and evaluation framework
- PyTorch team for the excellent deep learning framework
- VGG authors for the classic architecture

**Important:** This project builds upon the DomainBed framework but implements an independent experiment system. While we use DomainBed's datasets, all training code, model implementations, and experimental framework are original contributions.

## 📜 About This Project

This project started as an exploration of domain adaptation techniques using the DomainBed benchmark. While it uses DomainBed's datasets, it features:

- **Independent training framework**: Custom VGG-16 implementation with adaptive architecture
- **Original experiment management**: Comprehensive logging, visualization, and batch experiment system  
- **Modular design**: Clean separation of concerns for easy extension and modification
- **Enhanced analysis tools**: Rich result visualization and performance comparison capabilities

The goal is to provide a user-friendly, well-documented framework for domain adaptation research that complements the DomainBed ecosystem.

## 📊 Recent Experiment Results

| Dataset | Test Env | Best Accuracy | Configuration | Training Time |
|---------|----------|---------------|---------------|---------------|
| ColoredMNIST | 0 | 72.67% | lr=0.0001, bs=16, epochs=2 | ~84s |
| ColoredMNIST | 1 | 50.39% | lr=0.0001, bs=32, epochs=2 | ~52s |
| TerraIncognita | 0 | 44.07% | lr=0.0001, bs=16, epochs=1 | ~537s |

*Results may vary based on hardware and random initialization**
