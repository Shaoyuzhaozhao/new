"""
评估脚本

实现模型评估和结果分析
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm

from .models import get_model
from .data import create_data_loaders, create_cross_environment_loaders
from .utils import compute_all_metrics, plot_confusion_matrix
from .preset import preset, ACTIVITY_LABELS


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    Returns:
        评估结果字典
    """
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    avg_loss = total_loss / len(test_loader.dataset)
    
    # 计算指标
    num_classes = len(np.unique(all_targets))
    class_names = [ACTIVITY_LABELS.get(i, str(i)) for i in range(num_classes)]
    metrics = compute_all_metrics(all_targets, all_preds, class_names)
    
    results = {
        'loss': avg_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision_macro'],
        'recall': metrics['recall_macro'],
        'f1': metrics['f1_macro'],
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'per_class_accuracy': (metrics['confusion_matrix'].diagonal() / 
                               metrics['confusion_matrix'].sum(axis=1)).tolist(),
    }
    
    return results


def load_and_evaluate(
    model_path: str,
    model_name: str,
    config: dict,
    environment: Optional[List[str]] = None,
) -> Dict:
    """
    加载模型并评估
    
    Args:
        model_path: 模型文件路径
        model_name: 模型名称
        config: 配置字典
        environment: 环境列表
        
    Returns:
        评估结果
    """
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据加载器
    data_config = config.get('data', {})
    env = environment if environment else data_config.get('environment', ['classroom'])
    
    _, test_loader = create_data_loaders(
        data_dir=data_config.get('path_data_x', 'dataset/wifi_csi/amp'),
        annotation_path=data_config.get('path_data_y', 'dataset/annotation.csv'),
        task=config.get('task', {}).get('name', 'activity'),
        num_users=data_config.get('num_users', ['5']),
        wifi_band=data_config.get('wifi_band', ['2.4']),
        environment=env,
        batch_size=config.get('train', {}).get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        augmentation_config=None,  # 测试时不使用增强
    )
    
    # 创建并加载模型
    num_classes = config.get('task', {}).get('num_classes', {}).get('activity', 9)
    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # 评估
    results = evaluate_model(model, test_loader, device)
    results['model'] = model_name
    results['environment'] = env
    
    return results


def run_all_evaluations(
    config: dict,
    models: List[str] = ['SDAN', 'AlexNet', 'ResNet18', 'ResNet_TS', 'RF_Net'],
    environments: List[str] = ['classroom', 'meeting_room', 'empty_room'],
    save_results: bool = True,
) -> Dict:
    """
    运行所有评估实验
    
    Args:
        config: 配置字典
        models: 模型列表
        environments: 环境列表
        save_results: 是否保存结果
        
    Returns:
        所有结果的字典
    """
    all_results = {}
    
    for model_name in models:
        model_results = {}
        
        for env in environments:
            print(f"\nEvaluating {model_name} on {env}...")
            
            model_path = os.path.join(
                config.get('output', {}).get('path_model', 'results/models'),
                f'{model_name}_all_best.pth'
            )
            
            if os.path.exists(model_path):
                results = load_and_evaluate(
                    model_path, model_name, config, environment=[env]
                )
                model_results[env] = results['accuracy']
            else:
                print(f"Model not found: {model_path}")
                model_results[env] = 0.0
        
        all_results[model_name] = model_results
    
    if save_results:
        save_path = os.path.join(
            config.get('output', {}).get('path_result', 'results'),
            'evaluation_results.json'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {save_path}")
    
    return all_results


def run_cross_environment_evaluation(
    config: dict,
    models: List[str] = ['SDAN', 'ResNet18', 'RF_Net'],
    environments: List[str] = ['classroom', 'meeting_room', 'empty_room'],
) -> Dict:
    """
    运行跨环境评估实验
    
    Args:
        config: 配置字典
        models: 模型列表
        environments: 环境列表
        
    Returns:
        跨环境评估结果
    """
    results = {}
    
    for model_name in models:
        model_results = []
        
        for train_env in environments:
            for test_env in environments:
                if train_env != test_env:
                    print(f"\n{model_name}: {train_env} -> {test_env}")
                    
                    # 这里假设已经训练好了跨环境模型
                    model_path = os.path.join(
                        config.get('output', {}).get('path_model', 'results/models'),
                        f'{model_name}_{train_env}_to_{test_env}.pth'
                    )
                    
                    if os.path.exists(model_path):
                        eval_results = load_and_evaluate(
                            model_path, model_name, config, environment=[test_env]
                        )
                        model_results.append({
                            'train_env': train_env,
                            'test_env': test_env,
                            'accuracy': eval_results['accuracy'],
                        })
        
        results[model_name] = model_results
    
    return results


def run_ablation_study(
    config: dict,
    model_name: str = 'SDAN',
) -> Dict:
    """
    运行消融实验
    
    Args:
        config: 配置字典
        model_name: 基础模型名称
        
    Returns:
        消融实验结果
    """
    ablation_configs = {
        'Full SDAN': 'SDAN',
        'w/o Multi-scale Feature': 'SDAN_no_multiscale',
        'w/o Time-Freq Decoupling': 'SDAN_no_tfdecoupling',
        'w/o Dynamic Attention': 'SDAN_no_attention',
        'Replace with ResNet-18': 'ResNet18',
    }
    
    results = {}
    
    for name, model_variant in ablation_configs.items():
        print(f"\nEvaluating {name}...")
        
        model_path = os.path.join(
            config.get('output', {}).get('path_model', 'results/models'),
            f'{model_variant}_all_best.pth'
        )
        
        if os.path.exists(model_path):
            eval_results = load_and_evaluate(model_path, model_variant, config)
            results[name] = eval_results['accuracy']
        else:
            print(f"Model not found: {model_path}")
            results[name] = 0.0
    
    return results


def run_augmentation_ablation(
    config: dict,
    model_name: str = 'SDAN',
) -> Dict:
    """
    运行数据增强消融实验
    
    Args:
        config: 配置字典
        model_name: 模型名称
        
    Returns:
        增强消融结果
    """
    augmentation_configs = {
        'Baseline (No Aug)': 'none',
        '+FDA': 'fda',
        '+TDA': 'tda',
        '+MDA': 'mda',
        '+FDA+TDA': 'fda_tda',
        '+FDA+MDA': 'fda_mda',
        '+TDA+MDA': 'tda_mda',
        'Full (All)': 'all',
    }
    
    results = {}
    
    for name, aug_mode in augmentation_configs.items():
        print(f"\nEvaluating {name}...")
        
        model_path = os.path.join(
            config.get('output', {}).get('path_model', 'results/models'),
            f'{model_name}_{aug_mode}_best.pth'
        )
        
        if os.path.exists(model_path):
            eval_results = load_and_evaluate(model_path, model_name, config)
            results[name] = eval_results['accuracy']
        else:
            print(f"Model not found: {model_path}")
            results[name] = 0.0
    
    return results


if __name__ == '__main__':
    # 运行所有评估
    results = run_all_evaluations(preset)
    print("\nAll evaluation results:")
    print(json.dumps(results, indent=2))
