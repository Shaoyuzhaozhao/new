"""
频谱动态注意力网络 (SDAN)

实现论文3.3节的完整SDAN网络架构:
- 多尺度特征提取模块
- 时频解耦建模模块
- 动态通道注意力模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .layers import (
    MultiScaleConvBlock,
    TimeFrequencyDecoupling,
    DynamicChannelAttention,
)


class SDAN(nn.Module):
    """
    频谱动态注意力网络 (Spectrogram-centric Dynamic Attention Network)
    
    专门为WiFi时频频谱图设计的网络架构，包含三个核心模块:
    1. 多尺度特征提取模块：捕获不同粒度的时频特征
    2. 时频解耦建模模块：分别学习时间演化和频率分布
    3. 动态通道注意力模块：自适应调整特征权重
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
        multi_scale_channels: List[int] = [32, 32, 32, 32],
        kernel_sizes: List[int] = [3, 5, 7],
        temporal_kernel: int = 7,
        frequency_kernel: int = 7,
        dropout: float = 0.5,
    ):
        """
        初始化SDAN网络
        
        Args:
            input_channels: 输入通道数（频谱图通常为1）
            num_classes: 分类类别数
            multi_scale_channels: 多尺度卷积各分支通道数
            kernel_sizes: 多尺度卷积核大小
            temporal_kernel: 时间卷积核大小
            frequency_kernel: 频率卷积核大小
            dropout: Dropout比例
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 输入标准化层
        self.input_norm = nn.BatchNorm2d(input_channels)
        
        # 1. 多尺度特征提取模块
        self.multi_scale = MultiScaleConvBlock(
            in_channels=input_channels,
            out_channels=multi_scale_channels[0],
            kernel_sizes=kernel_sizes,
        )
        
        # 多尺度输出通道数 = 32 * 4 = 128
        multi_scale_out = multi_scale_channels[0] * (len(kernel_sizes) + 1)
        
        # 2. 时频解耦建模模块
        self.time_freq_decoupling = TimeFrequencyDecoupling(
            in_channels=multi_scale_out,
            out_channels=multi_scale_out // 2,
            temporal_kernel=temporal_kernel,
            frequency_kernel=frequency_kernel,
        )
        
        # 3. 动态通道注意力模块
        self.dynamic_attention = DynamicChannelAttention(
            num_channels=multi_scale_out,
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(multi_scale_out, multi_scale_out // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(multi_scale_out // 2, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入频谱图，形状为 (B, C, F, T)
               B: batch size
               C: 通道数（通常为1）
               F: 频率bins数
               T: 时间帧数

        Returns:
            分类输出，形状为 (B, num_classes)
        """
        # 输入标准化
        x = self.input_norm(x)

        # 多尺度特征提取
        x = self.multi_scale(x)  # (B, 128, F, T)

        # 时频解耦建模
        x = self.time_freq_decoupling(x)  # (B, 128, F, T)

        # 动态通道注意力
        x = self.dynamic_attention(x)  # (B, 128, F, T)

        # 全局平均池化
        x = self.global_avg_pool(x)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)

        # 分类
        x = self.classifier(x)  # (B, num_classes)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征表示（用于可视化和分析）

        Args:
            x: 输入频谱图

        Returns:
            特征向量
        """
        x = self.input_norm(x)
        x = self.multi_scale(x)
        x = self.time_freq_decoupling(x)
        x = self.dynamic_attention(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class SDANWithoutMultiScale(nn.Module):
    """
    消融实验：移除多尺度特征提取模块的SDAN
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
        hidden_channels: int = 128,
        temporal_kernel: int = 7,
        frequency_kernel: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.input_norm = nn.BatchNorm2d(input_channels)

        # 使用单一尺度卷积替代
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.time_freq_decoupling = TimeFrequencyDecoupling(
            in_channels=hidden_channels,
            out_channels=hidden_channels // 2,
            temporal_kernel=temporal_kernel,
            frequency_kernel=frequency_kernel,
        )

        self.dynamic_attention = DynamicChannelAttention(num_channels=hidden_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.conv(x)
        x = self.time_freq_decoupling(x)
        x = self.dynamic_attention(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.classifier(x)


class SDANWithoutTFDecoupling(nn.Module):
    """
    消融实验：移除时频解耦模块的SDAN
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
        multi_scale_channels: List[int] = [32, 32, 32, 32],
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.5,
    ):
        super().__init__()

        self.input_norm = nn.BatchNorm2d(input_channels)

        self.multi_scale = MultiScaleConvBlock(
            in_channels=input_channels,
            out_channels=multi_scale_channels[0],
            kernel_sizes=kernel_sizes,
        )

        multi_scale_out = multi_scale_channels[0] * (len(kernel_sizes) + 1)

        # 使用标准2D卷积替代时频解耦
        self.conv = nn.Sequential(
            nn.Conv2d(multi_scale_out, multi_scale_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(multi_scale_out),
            nn.ReLU(inplace=True),
        )

        self.dynamic_attention = DynamicChannelAttention(num_channels=multi_scale_out)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(multi_scale_out, multi_scale_out // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(multi_scale_out // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.multi_scale(x)
        x = self.conv(x)
        x = self.dynamic_attention(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.classifier(x)


class SDANWithoutAttention(nn.Module):
    """
    消融实验：移除动态注意力模块的SDAN
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
        multi_scale_channels: List[int] = [32, 32, 32, 32],
        kernel_sizes: List[int] = [3, 5, 7],
        temporal_kernel: int = 7,
        frequency_kernel: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.input_norm = nn.BatchNorm2d(input_channels)

        self.multi_scale = MultiScaleConvBlock(
            in_channels=input_channels,
            out_channels=multi_scale_channels[0],
            kernel_sizes=kernel_sizes,
        )

        multi_scale_out = multi_scale_channels[0] * (len(kernel_sizes) + 1)

        self.time_freq_decoupling = TimeFrequencyDecoupling(
            in_channels=multi_scale_out,
            out_channels=multi_scale_out // 2,
            temporal_kernel=temporal_kernel,
            frequency_kernel=frequency_kernel,
        )

        # 不使用注意力模块
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(multi_scale_out, multi_scale_out // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(multi_scale_out // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.multi_scale(x)
        x = self.time_freq_decoupling(x)
        x = self.global_avg_pool(x).view(x.size(0), -1)
        return self.classifier(x)


def create_sdan(
    num_classes: int = 9,
    ablation: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    创建SDAN模型

    Args:
        num_classes: 类别数
        ablation: 消融实验类型
            - None: 完整SDAN
            - "no_multiscale": 移除多尺度模块
            - "no_tfdecoupling": 移除时频解耦模块
            - "no_attention": 移除注意力模块
        **kwargs: 其他参数

    Returns:
        SDAN模型
    """
    if ablation is None:
        return SDAN(num_classes=num_classes, **kwargs)
    elif ablation == "no_multiscale":
        return SDANWithoutMultiScale(num_classes=num_classes, **kwargs)
    elif ablation == "no_tfdecoupling":
        return SDANWithoutTFDecoupling(num_classes=num_classes, **kwargs)
    elif ablation == "no_attention":
        return SDANWithoutAttention(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown ablation type: {ablation}")