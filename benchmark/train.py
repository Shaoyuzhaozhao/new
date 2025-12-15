"""
训练脚本
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from .models import get_model
from .utils import AverageMeter, compute_all_metrics, ExperimentLogger
from .utils import plot_training_curves, plot_confusion_matrix
from .preset import preset, ACTIVITY_LABELS


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device(config: dict) -> torch.device:
    """获取计算设备"""
    if config.get('device', {}).get('gpu', True) and torch.cuda.is_available():
        return torch.device(f"cuda:{config.get('device', {}).get('gpu_id', 0)}")
    return torch.device('cpu')


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    
    for data, target in tqdm(train_loader, desc='Training', leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean().item()
        loss_meter.update(loss.item(), data.size(0))
        acc_meter.update(acc, data.size(0))
    
    return loss_meter.avg, acc_meter.avg


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating', leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            loss_meter.update(loss.item(), data.size(0))
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = (all_preds == all_targets).mean()
    
    return loss_meter.avg, accuracy, all_targets, all_preds


def train(config, model_name='SDAN', augmentation_mode='all', 
          environment=None, save_model=True):
    """完整训练流程"""
    from .data import create_data_loaders
    
    set_seed(config.get('seed', 42))
    device = get_device(config)
    print(f"Using device: {device}")
    
    # 配置数据增强
    aug_config = None
    if augmentation_mode != 'none':
        aug_config = config.get('augmentation', {}).copy()
        aug_config['enabled'] = True

        # 解析增强模式
        if augmentation_mode == 'all':
            # 全部启用
            pass
        elif augmentation_mode in ['fda', 'tda', 'mda']:
            # 单独启用一个
            for mode in ['fda', 'tda', 'mda']:
                aug_config[mode] = {'enabled': mode == augmentation_mode}
        elif '_' in augmentation_mode:
            # 组合模式: fda_tda, fda_mda, tda_mda
            enabled_modes = augmentation_mode.split('_')
            for mode in ['fda', 'tda', 'mda']:
                aug_config[mode] = {'enabled': mode in enabled_modes}

    # 创建数据加载器
    data_config = config.get('data', {})
    env = environment or data_config.get('environment', ['classroom'])

    train_loader, test_loader = create_data_loaders(
        data_dir=data_config.get('path_data_x', 'dataset/wifi_csi/amp'),
        annotation_path=data_config.get('path_data_y', 'dataset/annotation.csv'),
        task=config.get('task', {}).get('name', 'activity'),
        num_users=data_config.get('num_users', ['5']),
        wifi_band=data_config.get('wifi_band', ['2.4']),
        environment=env,
        batch_size=config.get('train', {}).get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        augmentation_config=aug_config,
        train_ratio=data_config.get('train_ratio', 0.8),
        random_seed=config.get('seed', 42),
    )

    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 创建模型
    num_classes = config.get('task', {}).get('num_classes', {}).get('activity', 9)
    dropout = config.get('train', {}).get('dropout', 0.3)
    model = get_model(model_name, num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    print(f"Model: {model_name}, Params: {sum(p.numel() for p in model.parameters())}")

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    train_config = config.get('train', {})
    optimizer = optim.Adam(model.parameters(), lr=train_config.get('learning_rate', 1e-3),
                           weight_decay=train_config.get('weight_decay', 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.get('epochs', 100),
                                   eta_min=train_config.get('min_lr', 1e-6))

    # 日志
    logger = ExperimentLogger(f"{model_name}_{augmentation_mode}",
                              config.get('output', {}).get('path_log', 'results/logs'))
    logger.log_config(config)

    # 训练循环
    epochs = train_config.get('epochs', 100)
    best_acc, best_epoch = 0, 0
    patience_counter, early_stopping = 0, train_config.get('early_stopping', 20)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            patience_counter = 0
            if save_model:
                model_dir = config.get('output', {}).get('path_model', 'results/models')
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(),
                          os.path.join(model_dir, f'{model_name}_{augmentation_mode}_best.pth'))
        else:
            patience_counter += 1

        if patience_counter >= early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # 最终评估
    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    class_names = [ACTIVITY_LABELS[i] for i in range(num_classes)]
    metrics = compute_all_metrics(y_true, y_pred, class_names)

    results = {
        'model': model_name, 'augmentation': augmentation_mode, 'environment': env,
        'best_epoch': best_epoch, 'best_val_acc': best_acc, 'final_acc': final_acc,
        'metrics': {'accuracy': metrics['accuracy'], 'precision': metrics['precision_macro'],
                    'recall': metrics['recall_macro'], 'f1': metrics['f1_macro']},
    }

    logger.log_results(results)
    logger.save_history()

    # 绘图
    fig_dir = config.get('output', {}).get('path_figure', 'visualize')
    plot_training_curves(logger.history,
                        save_path=os.path.join(fig_dir, f'{model_name}_training.png'))
    plot_confusion_matrix(metrics['confusion_matrix'], class_names,
                         save_path=os.path.join(fig_dir, f'{model_name}_confusion.png'))

    print(f"\nBest acc: {best_acc:.4f} (epoch {best_epoch}), Final: {final_acc:.4f}")
    return results


def train_cross_environment(config, model_name='SDAN', train_env='classroom',
                            test_env='meeting_room', augmentation_mode='all'):
    """
    跨环境训练和测试

    在一个环境训练，在另一个环境测试
    """
    from .data import create_cross_environment_loaders

    set_seed(config.get('seed', 42))
    device = get_device(config)
    print(f"Using device: {device}")
    print(f"Train env: {train_env}, Test env: {test_env}")

    # 配置数据增强
    aug_config = None
    if augmentation_mode != 'none':
        aug_config = config.get('augmentation', {}).copy()
        aug_config['enabled'] = True

    # 创建数据加载器
    data_config = config.get('data', {})

    train_loader, test_loader = create_cross_environment_loaders(
        data_dir=data_config.get('path_data_x', 'dataset/wifi_csi/amp'),
        annotation_path=data_config.get('path_data_y', 'dataset/annotation.csv'),
        train_env=train_env,
        test_env=test_env,
        task=config.get('task', {}).get('name', 'activity'),
        num_users=data_config.get('num_users', ['5']),
        wifi_band=data_config.get('wifi_band', ['2.4']),
        batch_size=config.get('train', {}).get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        augmentation_config=aug_config,
    )

    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 创建模型
    num_classes = config.get('task', {}).get('num_classes', {}).get('activity', 9)
    model = get_model(model_name, num_classes=num_classes)
    model = model.to(device)

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    train_config = config.get('train', {})
    optimizer = optim.Adam(model.parameters(), lr=train_config.get('learning_rate', 1e-3),
                           weight_decay=train_config.get('weight_decay', 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.get('epochs', 100),
                                   eta_min=train_config.get('min_lr', 1e-6))

    # 训练循环
    epochs = train_config.get('epochs', 100)
    best_acc, best_epoch = 0, 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, test_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch

    # 最终评估
    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    results = {
        'model': model_name,
        'train_env': train_env,
        'test_env': test_env,
        'best_epoch': best_epoch,
        'best_val_acc': best_acc,
        'final_acc': final_acc,
    }

    print(f"\nCross-env result: {train_env} -> {test_env}: {final_acc:.4f}")
    return results


if __name__ == '__main__':
    results = train(preset, model_name='SDAN', augmentation_mode='all')
    print(results)