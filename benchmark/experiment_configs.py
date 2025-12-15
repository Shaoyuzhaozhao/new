"""
实验配置文件

包含论文中所有实验的完整配置参数
"""

# =====================================================
# 论文实验配置
# =====================================================

# 基础配置（从preset.py继承）
from .preset import preset

# 论文表1: 性能对比实验配置
EXPERIMENT_PERFORMANCE_COMPARISON = {
    'models': ['SDAN', 'AlexNet', 'ResNet18', 'ResNet_TS', 'RF_Net'],
    'environments': ['classroom', 'meeting_room', 'empty_room'],
    'num_users': ['5'],  # 5用户场景
    'wifi_band': ['2.4'],
    'augmentation': 'all',  # 完整物理数据增强
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'repeat': 5,  # 5次重复实验
}

# 论文图9: 数据增强消融实验配置
EXPERIMENT_AUGMENTATION_ABLATION = {
    'model': 'SDAN',
    'augmentation_modes': [
        ('none', 'Baseline'),
        ('fda', '+FDA'),
        ('tda', '+TDA'),
        ('mda', '+MDA'),
        ('fda_tda', '+FDA+TDA'),
        ('fda_mda', '+FDA+MDA'),
        ('tda_mda', '+TDA+MDA'),
        ('all', 'Full (All)'),
    ],
    'epochs': 100,
    'repeat': 3,
}

# 论文图10: 物理增强 vs 小波去噪配置
EXPERIMENT_AUG_VS_DENOISE = {
    'models': ['SDAN', 'ResNet18', 'AlexNet', 'ResNet_TS', 'RF_Net'],
    'preprocessing_modes': [
        'none',           # 无预处理
        'wavelet',        # 小波去噪
        'physical_aug',   # 物理数据增强
    ],
    'epochs': 100,
    'repeat': 3,
}

# 论文图11: SDAN模块消融实验配置
EXPERIMENT_MODULE_ABLATION = {
    'ablation_configs': [
        ('SDAN', None, 'Full SDAN'),
        ('SDAN', 'no_multiscale', 'w/o Multi-scale Feature'),
        ('SDAN', 'no_tfdecoupling', 'w/o Time-Freq Decoupling'),
        ('SDAN', 'no_attention', 'w/o Dynamic Attention'),
        ('ResNet18', None, 'Replace with ResNet-18'),
    ],
    'epochs': 100,
    'repeat': 3,
}

# 论文表2: 跨环境泛化实验配置
EXPERIMENT_CROSS_ENVIRONMENT = {
    'model': 'SDAN',
    'train_environments': ['classroom', 'meeting_room', 'empty_room'],
    'test_environments': ['classroom', 'meeting_room', 'empty_room'],
    'augmentation': 'all',
    'epochs': 100,
    'repeat': 3,
}

# 论文表3: 多任务实验配置
EXPERIMENT_MULTI_TASK = {
    'model': 'SDAN',
    'tasks': ['activity', 'identity', 'location'],
    'num_classes': {
        'activity': 9,
        'identity': 6,
        'location': 15,
    },
    'epochs': 100,
    'repeat': 3,
}

# 不同用户数量实验配置
EXPERIMENT_USER_NUMBERS = {
    'model': 'SDAN',
    'num_users_list': ['0', '1', '2', '3', '4', '5'],
    'augmentation': 'all',
    'epochs': 100,
    'repeat': 3,
}

# =====================================================
# 论文中的预期结果（用于验证）
# =====================================================

EXPECTED_RESULTS = {
    # 表1: 性能对比 (平均准确率%)
    'performance_comparison': {
        'SDAN': 77.9,  # 论文摘要提到~80%
        'ResNet-18': 69.8,
        'AlexNet': 69.5,
        'ResNet-TS': 67.4,
        'RF-Net': 58.4,
    },
    
    # 图9: 数据增强消融
    'augmentation_ablation': {
        'Baseline': 68.5,
        '+FDA': 72.5,
        '+TDA': 71.8,
        '+MDA': 73.2,
        'Full': 77.9,
    },
    
    # 跨环境泛化性能下降
    'cross_environment_drop': 1.7,  # SDAN仅下降1.7%
    
    # 模块贡献
    'module_contribution': {
        'Multi-scale': 3.6,
        'Time-Freq Decoupling': 2.4,
        'Dynamic Attention': 1.6,
    },
}

# =====================================================
# 数据增强具体参数（论文3.2节）
# =====================================================

FDA_CONFIG = {
    'num_bands': 6,  # ISS-N中的N=6
    'motion_threshold': 0.5,
}

TDA_CONFIG = {
    'base_window': 256,
    'scales': [0.5, 1.0, 2.0],  # 短/标准/长窗口
    'hop_length': 64,
}

MDA_CONFIG = {
    'erase_ratio': 0.2,  # 运动区域边缘擦除20%
    'static_erase_ratio': 0.5,  # 静态区域擦除50%
    'energy_threshold': 0.7,
    'shift_range': 0.3,
}

# =====================================================
# SDAN网络参数（论文3.3节）
# =====================================================

SDAN_CONFIG = {
    'input_channels': 1,
    'num_classes': 9,
    'multi_scale_channels': [32, 32, 32, 32],  # 4个分支各32通道
    'kernel_sizes': [3, 5, 7],  # 多尺度卷积核
    'temporal_kernel': 7,  # 时间卷积核Kt
    'frequency_kernel': 7,  # 频率卷积核Kf
    'dropout': 0.5,
}

# =====================================================
# STFT参数（论文3.1节）
# =====================================================

STFT_CONFIG = {
    'n_fft': 256,
    'hop_length': 64,
    'window': 'hann',
    'normalized': True,
}

# =====================================================
# 活动类别（论文数据集部分）
# =====================================================

ACTIVITY_CLASSES = {
    0: 'No Action',
    1: 'Walk',
    2: 'Rotate',
    3: 'Jump',
    4: 'Wave',
    5: 'Lie Down',
    6: 'Pick Up',
    7: 'Sit Down',
    8: 'Stand Up',
}

# 环境映射
ENVIRONMENT_CLASSES = {
    'classroom': 0,
    'meeting_room': 1,
    'empty_room': 2,
}
