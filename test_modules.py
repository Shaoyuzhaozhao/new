#!/usr/bin/env python
"""
测试脚本 - 验证所有模块可以正常导入和基本功能正常

运行: python test_modules.py
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("="*60)
print("WiFi多用户感知系统 - 模块测试")
print("="*60)

# 测试1: 导入所有模块
print("\n[1] 测试模块导入...")
try:
    from benchmark.preset import preset, ACTIVITY_LABELS
    print("  ✓ preset模块导入成功")
    
    from benchmark.data.spectrogram import (
        SpectrogramTransform, 
        CSIToSpectrogram,
        compute_motion_statistics,
        select_motion_sensitive_subcarriers,
    )
    print("  ✓ spectrogram模块导入成功")
    
    from benchmark.data.augmentation import (
        FrequencyDomainAugmentation,
        TemporalDomainAugmentation,
        MotionAwareAugmentation,
        PhysicalDataAugmentation,
    )
    print("  ✓ augmentation模块导入成功")
    
    from benchmark.models.sdan import SDAN, create_sdan
    from benchmark.models.baselines import AlexNet, ResNet18, ResNetTS, RFNet
    from benchmark.models import get_model
    print("  ✓ models模块导入成功")
    
    from benchmark.utils import (
        compute_all_metrics,
        plot_confusion_matrix,
        ExperimentLogger,
    )
    print("  ✓ utils模块导入成功")
    
except ImportError as e:
    print(f"  ✗ 导入错误: {e}")
    sys.exit(1)

# 测试2: 频谱图转换
print("\n[2] 测试频谱图转换...")
try:
    # 创建模拟CSI数据
    T = 3000  # 3秒 * 1000Hz采样率
    K = 30    # 子载波数
    
    csi_signal = np.random.randn(T) + 1j * np.random.randn(T)
    csi_signal = np.abs(csi_signal)  # 取幅度
    
    # STFT变换
    stft = SpectrogramTransform(n_fft=256, hop_length=64)
    spectrogram = stft.transform(csi_signal, return_numpy=True)
    
    print(f"  输入CSI形状: {csi_signal.shape}")
    print(f"  输出频谱图形状: {spectrogram.shape}")
    print("  ✓ 频谱图转换成功")
    
except Exception as e:
    print(f"  ✗ 频谱图转换错误: {e}")

# 测试3: 运动统计量计算
print("\n[3] 测试运动统计量计算...")
try:
    ms = compute_motion_statistics(csi_signal)
    print(f"  运动统计量: {ms:.4f}")
    print("  ✓ 运动统计量计算成功")
except Exception as e:
    print(f"  ✗ 运动统计量计算错误: {e}")

# 测试4: 数据增强
print("\n[4] 测试数据增强...")
try:
    # FDA
    csi_2d = np.random.randn(T, K)
    fda = FrequencyDomainAugmentation(num_bands=6)
    fda_specs = fda(csi_2d)
    print(f"  FDA生成 {len(fda_specs)} 个频谱图")
    
    # TDA
    tda = TemporalDomainAugmentation(scales=[0.5, 1.0, 2.0])
    tda_specs = tda(csi_signal)
    print(f"  TDA生成 {len(tda_specs)} 个频谱图")
    
    # MDA
    mda = MotionAwareAugmentation()
    mda_spec = mda(spectrogram)
    print(f"  MDA输出形状: {mda_spec.shape}")
    
    print("  ✓ 数据增强测试成功")
except Exception as e:
    print(f"  ✗ 数据增强错误: {e}")

# 测试5: SDAN网络
print("\n[5] 测试SDAN网络...")
try:
    # 创建SDAN模型
    model = SDAN(input_channels=1, num_classes=9)
    
    # 创建模拟输入
    batch_size = 4
    F_bins = 129  # (n_fft/2 + 1)
    T_frames = 47  # 根据hop_length
    
    x = torch.randn(batch_size, 1, F_bins, T_frames)
    
    # 前向传播
    output = model(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("  ✓ SDAN网络测试成功")
    
except Exception as e:
    print(f"  ✗ SDAN网络错误: {e}")

# 测试6: 基线模型
print("\n[6] 测试基线模型...")
try:
    models_to_test = ['AlexNet', 'ResNet18', 'ResNet_TS', 'RF_Net']
    
    for model_name in models_to_test:
        model = get_model(model_name, num_classes=9)
        output = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name}: 输出{output.shape}, 参数{params:,}")
    
    print("  ✓ 基线模型测试成功")
except Exception as e:
    print(f"  ✗ 基线模型错误: {e}")

# 测试7: 消融模型
print("\n[7] 测试消融模型...")
try:
    ablation_types = [None, 'no_multiscale', 'no_tfdecoupling', 'no_attention']
    
    for ablation in ablation_types:
        model = create_sdan(num_classes=9, ablation=ablation)
        output = model(x)
        name = f"SDAN_{ablation}" if ablation else "Full SDAN"
        print(f"  {name}: 输出{output.shape}")
    
    print("  ✓ 消融模型测试成功")
except Exception as e:
    print(f"  ✗ 消融模型错误: {e}")

# 测试8: 指标计算
print("\n[8] 测试指标计算...")
try:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])
    
    metrics = compute_all_metrics(y_true, y_pred)
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision_macro']:.4f}")
    print(f"  召回率: {metrics['recall_macro']:.4f}")
    print(f"  F1分数: {metrics['f1_macro']:.4f}")
    print("  ✓ 指标计算测试成功")
except Exception as e:
    print(f"  ✗ 指标计算错误: {e}")

print("\n" + "="*60)
print("所有测试完成!")
print("="*60)

# 打印配置摘要
print("\n配置摘要:")
print(f"  活动类别: {list(ACTIVITY_LABELS.values())}")
print(f"  默认用户数: {preset['data']['num_users']}")
print(f"  默认环境: {preset['data']['environment']}")
print(f"  训练轮数: {preset['train']['epochs']}")
print(f"  批大小: {preset['train']['batch_size']}")
print(f"  学习率: {preset['train']['learning_rate']}")
