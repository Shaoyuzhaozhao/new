"""
评估指标模块
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算准确率"""
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """计算精确率"""
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """计算召回率"""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """计算F1分数"""
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """计算混淆矩阵"""
    return confusion_matrix(y_true, y_pred)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    计算所有评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        
    Returns:
        包含各项指标的字典
    """
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision_macro': compute_precision(y_true, y_pred, 'macro'),
        'recall_macro': compute_recall(y_true, y_pred, 'macro'),
        'f1_macro': compute_f1(y_true, y_pred, 'macro'),
        'precision_weighted': compute_precision(y_true, y_pred, 'weighted'),
        'recall_weighted': compute_recall(y_true, y_pred, 'weighted'),
        'f1_weighted': compute_f1(y_true, y_pred, 'weighted'),
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred),
    }
    
    # 每类指标
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class_precision'] = per_class_precision
    metrics['per_class_recall'] = per_class_recall
    metrics['per_class_f1'] = per_class_f1
    
    if class_names is not None:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
    
    return metrics


class AverageMeter:
    """
    用于跟踪和计算平均值的工具类
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """
    用于跟踪多个指标的工具类
    """
    
    def __init__(self, metric_names: List[str]):
        self.metrics = {name: AverageMeter() for name in metric_names}
    
    def reset(self):
        for meter in self.metrics.values():
            meter.reset()
    
    def update(self, metric_name: str, val: float, n: int = 1):
        if metric_name in self.metrics:
            self.metrics[metric_name].update(val, n)
    
    def get_avg(self, metric_name: str) -> float:
        if metric_name in self.metrics:
            return self.metrics[metric_name].avg
        return 0.0
    
    def get_all_avg(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.metrics.items()}
