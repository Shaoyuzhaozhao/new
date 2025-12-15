"""
基线模型模块

实现论文中用于对比的基线模型:
- AlexNet
- ResNet-18
- ResNet-TS (时间序列ResNet)
- RF-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .layers import ResidualBlock, SEBlock


class AlexNet(nn.Module):
    """
    AlexNet基线模型
    
    经典CNN架构，适配WiFi频谱图输入
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    """
    ResNet-18基线模型
    
    深度残差网络，18层配置
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
    ):
        super().__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResNetTS(nn.Module):
    """
    ResNet-TS基线模型
    
    针对时间序列设计的ResNet变体
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
        hidden_channels: List[int] = [64, 128, 128],
    ):
        super().__init__()
        
        layers = []
        in_ch = input_channels
        
        for out_ch in hidden_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RFNet(nn.Module):
    """
    RF-Net基线模型
    
    专门为WiFi感知设计的元学习框架
    简化版本：双路径网络结构
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 9,
        feature_dim: int = 128,
    ):
        super().__init__()
        
        # 特征提取路径1：卷积网络
        self.conv_path = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # 特征提取路径2：全局统计特征
        self.stat_path = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(input_channels * 64, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(128 + feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        # 分类器
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 路径1
        conv_feat = self.conv_path(x)
        conv_feat = torch.flatten(conv_feat, 1)
        
        # 路径2
        stat_feat = self.stat_path(x)
        
        # 融合
        fused = torch.cat([conv_feat, stat_feat], dim=1)
        fused = self.fusion(fused)
        
        # 分类
        out = self.classifier(fused)
        
        return out


class MLP(nn.Module):
    """简单MLP基线"""
    
    def __init__(
        self,
        input_size: int = 129 * 47,  # F * T
        hidden_sizes: List[int] = [512, 256],
        num_classes: int = 9,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        layers = []
        in_size = input_size
        
        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, h_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_size = h_size
        
        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.network(x)


class LSTM(nn.Module):
    """LSTM基线"""
    
    def __init__(
        self,
        input_size: int = 129,  # 频率bins
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 9,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) -> (B, T, F)
        if x.ndim == 4:
            x = x.squeeze(1).permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out


class CNN1D(nn.Module):
    """1D CNN基线"""
    
    def __init__(
        self,
        input_channels: int = 129,  # 频率bins
        hidden_channels: List[int] = [64, 128, 256],
        num_classes: int = 9,
    ):
        super().__init__()
        
        layers = []
        in_ch = input_channels
        
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T) -> (B, F, T)
        if x.ndim == 4:
            x = x.squeeze(1)
        
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_model(
    model_name: str,
    num_classes: int = 9,
    input_channels: int = 1,
    **kwargs
) -> nn.Module:
    """
    创建模型
    
    Args:
        model_name: 模型名称
        num_classes: 类别数
        input_channels: 输入通道数
        **kwargs: 其他参数
        
    Returns:
        模型实例
    """
    models = {
        'AlexNet': AlexNet,
        'ResNet18': ResNet18,
        'ResNet_TS': ResNetTS,
        'RF_Net': RFNet,
        'MLP': MLP,
        'LSTM': LSTM,
        'CNN1D': CNN1D,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    # 过滤掉不支持的参数
    supported_kwargs = {}
    import inspect
    sig = inspect.signature(models[model_name].__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    for k, v in kwargs.items():
        if k in valid_params:
            supported_kwargs[k] = v

    return models[model_name](
        input_channels=input_channels,
        num_classes=num_classes,
        **supported_kwargs
    )