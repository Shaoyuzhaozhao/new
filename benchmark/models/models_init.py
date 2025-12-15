"""
模型模块
"""

from .sdan import (
    SDAN,
    SDANWithoutMultiScale,
    SDANWithoutTFDecoupling,
    SDANWithoutAttention,
    create_sdan,
)

from .sdan_v2 import (
    SDANV2,
    create_sdan_v2,
    LabelSmoothingCrossEntropy,
)

from .baselines import (
    AlexNet,
    ResNet18,
    ResNetTS,
    RFNet,
    MLP,
    LSTM,
    CNN1D,
    create_model,
)

from .layers import (
    MultiScaleConvBlock,
    TimeFrequencyDecoupling,
    DynamicChannelAttention,
    SEBlock,
    ResidualBlock,
)


def get_model(model_name: str, num_classes: int = 9, **kwargs):
    """
    获取模型

    Args:
        model_name: 模型名称
        num_classes: 类别数
        **kwargs: 其他参数

    Returns:
        模型实例
    """
    if model_name == 'SDAN':
        return create_sdan(num_classes=num_classes, **kwargs)
    elif model_name == 'SDAN_V2':
        return create_sdan_v2(num_classes=num_classes, **kwargs)
    elif model_name.startswith('SDAN_'):
        # 消融实验模型
        ablation = model_name.replace('SDAN_', '')
        return create_sdan(num_classes=num_classes, ablation=ablation, **kwargs)
    else:
        return create_model(model_name, num_classes=num_classes, **kwargs)


__all__ = [
    # SDAN
    'SDAN',
    'SDANWithoutMultiScale',
    'SDANWithoutTFDecoupling',
    'SDANWithoutAttention',
    'create_sdan',
    # SDAN V2
    'SDANV2',
    'create_sdan_v2',
    'LabelSmoothingCrossEntropy',
    # 基线模型
    'AlexNet',
    'ResNet18',
    'ResNetTS',
    'RFNet',
    'MLP',
    'LSTM',
    'CNN1D',
    'create_model',
    # 自定义层
    'MultiScaleConvBlock',
    'TimeFrequencyDecoupling',
    'DynamicChannelAttention',
    'SEBlock',
    'ResidualBlock',
    # 工厂函数
    'get_model',
]