"""
自定义网络层模块

包含SDAN网络中使用的各种自定义层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积模块
    
    论文3.3.1节：使用3×3、5×5、7×7并行卷积分支和池化路径
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        kernel_sizes: List[int] = [3, 5, 7],
        pool_size: int = 3,
    ):
        super().__init__()
        
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)
        
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.total_out_channels = out_channels * (len(kernel_sizes) + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        return torch.cat(outputs, dim=1)


class TimeFrequencyDecoupling(nn.Module):
    """
    时频解耦建模模块
    
    论文3.3.2节：分离时间演化和频率分布特征
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        temporal_kernel: int = 7,
        frequency_kernel: int = 7,
    ):
        super().__init__()
        
        self.temporal_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(frequency_kernel, 1),
                      padding=(frequency_kernel // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_feat = self.temporal_branch(x)
        frequency_feat = self.frequency_branch(x)
        concat_feat = torch.cat([temporal_feat, frequency_feat], dim=1)
        return self.fusion(concat_feat)


class DynamicChannelAttention(nn.Module):
    """
    动态通道注意力模块
    
    论文3.3.3节：自适应调整特征权重
    """
    
    def __init__(self, num_channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        
        t = int(abs(math.log2(num_channels) / gamma + b / gamma))
        k = t if t % 2 == 1 else t + 1
        k = max(k, 3)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        z = self.global_avg_pool(x).view(B, 1, C)
        z = self.conv1d(z)
        alpha = self.sigmoid(z).view(B, C, 1, 1)
        return x * alpha


class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)
