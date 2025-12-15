"""
数据处理模块
"""

from .spectrogram import (
    SpectrogramTransform,
    CSIToSpectrogram,
    compute_motion_statistics,
    select_motion_sensitive_subcarriers,
    detect_motion_regions,
)

from .augmentation import (
    FrequencyDomainAugmentation,
    TemporalDomainAugmentation,
    MotionAwareAugmentation,
    PhysicalDataAugmentation,
    SpectrogramAugmentationTransform,
)

from .dataset import (
    WiMANSDataset,
    WiMANSSpectrogramDataset,
    create_data_loaders,
    create_cross_environment_loaders,
)

__all__ = [
    # 频谱图转换
    'SpectrogramTransform',
    'CSIToSpectrogram',
    'compute_motion_statistics',
    'select_motion_sensitive_subcarriers',
    'detect_motion_regions',
    # 数据增强
    'FrequencyDomainAugmentation',
    'TemporalDomainAugmentation',
    'MotionAwareAugmentation',
    'PhysicalDataAugmentation',
    'SpectrogramAugmentationTransform',
    # 数据集
    'WiMANSDataset',
    'WiMANSSpectrogramDataset',
    'create_data_loaders',
    'create_cross_environment_loaders',
]
