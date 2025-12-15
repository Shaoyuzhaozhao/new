"""
日志工具模块
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logger(
    name: str,
    log_dir: str = "results/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        level: 日志级别
        
    Returns:
        日志器实例
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """
    实验日志记录器
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "results/logs",
        save_config: bool = True,
    ):
        """
        初始化实验日志记录器
        
        Args:
            experiment_name: 实验名称
            log_dir: 日志目录
            save_config: 是否保存配置
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建实验目录
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 设置日志器
        self.logger = setup_logger(experiment_name, self.experiment_dir)
        
        # 记录历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': [],
        }
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置"""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Config saved to {config_path}")
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
    ):
        """记录每个epoch的结果"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            msg += f", val_loss={val_loss:.4f}"
        
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
            msg += f", val_acc={val_acc:.4f}"
        
        self.logger.info(msg)
    
    def log_results(self, results: Dict[str, Any]):
        """记录最终结果"""
        results_path = os.path.join(self.experiment_dir, 'results.json')
        with open(results_path, 'w') as f:
            # 转换numpy数组为列表
            results_serializable = {}
            for k, v in results.items():
                if hasattr(v, 'tolist'):
                    results_serializable[k] = v.tolist()
                else:
                    results_serializable[k] = v
            json.dump(results_serializable, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.experiment_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def info(self, msg: str):
        """记录信息"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """记录警告"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """记录错误"""
        self.logger.error(msg)


def save_results(
    results: Dict[str, Any],
    save_path: str,
    append: bool = True,
):
    """
    保存实验结果
    
    Args:
        results: 结果字典
        save_path: 保存路径
        append: 是否追加到已有文件
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换为可序列化格式
    results_serializable = {}
    for k, v in results.items():
        if hasattr(v, 'tolist'):
            results_serializable[k] = v.tolist()
        else:
            results_serializable[k] = v
    
    # 加载已有结果
    existing_results = []
    if append and os.path.exists(save_path):
        with open(save_path, 'r') as f:
            existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
    
    # 追加新结果
    existing_results.append(results_serializable)
    
    # 保存
    with open(save_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
