# WiFi多用户感知系统：基于物理数据增强和频谱动态注意力网络

本项目实现了论文《基于物理数据增强和频谱动态注意力网络的WiFi多用户感知系统》中的方法。

## 项目结构

```
wifi_sensing_project/
├── benchmark/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sdan.py              # SDAN网络实现
│   │   ├── baselines.py         # 基线模型(AlexNet, ResNet等)
│   │   └── layers.py            # 自定义网络层
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # 数据集加载
│   │   ├── augmentation.py      # 物理数据增强
│   │   └── spectrogram.py       # CSI到频谱图转换
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py           # 评估指标
│   │   ├── visualization.py     # 可视化工具
│   │   └── logger.py            # 日志工具
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── preset.py                # 配置参数
├── configs/
│   └── default.yaml             # 默认配置
├── dataset/                     # 数据集目录(需下载WiMANS数据集)
├── results/                     # 实验结果保存目录
├── visualize/                   # 可视化输出目录
├── environment.yaml             # Conda环境配置
├── requirements.txt             # Python依赖
├── run_experiment.py            # 主实验运行脚本
└── README.md
```

## 环境配置

### 方法1: 使用Conda
```bash
conda env create -f environment.yaml
conda activate wifi_sensing
```

### 方法2: 使用pip
```bash
pip install -r requirements.txt
```

## 数据集准备

1. 从Kaggle下载WiMANS数据集: https://www.kaggle.com/datasets/shuokanghuang/wimans
2. 解压数据集到 `dataset/` 目录下
3. 确保目录结构如下:
```
dataset/
├── annotation.csv
├── wifi_csi/
│   ├── mat/         # 原始CSI数据
│   └── amp/         # 预处理的CSI幅度
└── video/           # 视频数据(可选)
```

## 运行实验

### 1. 完整实验
```bash
python run_experiment.py --model SDAN --task activity --augmentation all
```

### 2. 消融实验
```bash
# 不使用数据增强
python run_experiment.py --model SDAN --task activity --augmentation none

# 仅使用FDA
python run_experiment.py --model SDAN --task activity --augmentation fda

# 仅使用TDA
python run_experiment.py --model SDAN --task activity --augmentation tda

# 仅使用MDA
python run_experiment.py --model SDAN --task activity --augmentation mda
```

### 3. 基线模型对比
```bash
python run_experiment.py --model AlexNet --task activity
python run_experiment.py --model ResNet18 --task activity
python run_experiment.py --model ResNet_TS --task activity
python run_experiment.py --model RF_Net --task activity
```

### 4. 跨环境测试
```bash
python run_experiment.py --model SDAN --train_env classroom --test_env meeting_room
```

### 5. 生成论文图表
```bash
python benchmark/utils/visualization.py --generate_all
```

## 核心创新

### 1. 物理数据增强框架 (PDA)
- **FDA (频域增强)**: 智能子载波选择策略(ISS-N)，利用不同子载波对运动的差异化响应
- **TDA (时域增强)**: 多尺度时间窗口策略，使用0.5x、1x、2x三种窗口尺度
- **MDA (运动感知增强)**: 运动感知随机擦除(MRE)和随机平移(MRS)策略

### 2. 频谱动态注意力网络 (SDAN)
- **多尺度特征提取模块**: 3×3、5×5、7×7并行卷积分支 + 池化路径
- **时频解耦建模模块**: 分离时间演化和频率分布特征
- **动态通道注意力模块**: 自适应调整特征通道权重

## 实验结果

在WiMANS数据集5用户活动识别任务上:
- SDAN平均准确率: 77.9%
- 相比最优基线ResNet-18提升: 8.1%
- 跨环境泛化性能下降: 仅1.7%

## 引用

如果您使用本代码，请引用:

```bibtex
@article{wimans2024,
  title={WiMANS: A Benchmark Dataset for WiFi-based Multi-user Activity Sensing},
  author={Huang, Shuokang and others},
  journal={ECCV},
  year={2024}
}
```

## 许可证

MIT License
