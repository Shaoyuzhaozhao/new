"""
可视化工具模块

生成论文中的各种图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
):
    """
    绘制混淆矩阵 (论文图8)
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
        title: 标题
        figsize: 图像大小
        normalize: 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt='.2f' if normalize else 'd',
        cmap='Blues', xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    environments: List[str] = ["Empty Room", "Classroom", "Meeting Room"],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    绘制性能对比图 (论文图7)
    
    Args:
        results: 结果字典 {model_name: {env: accuracy}}
        environments: 环境列表
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, env in enumerate(environments):
        accuracies = [results[model].get(env, 0) for model in models]
        bars = ax.bar(x + i * width, accuracies, width, label=env, color=colors[i])
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Performance Comparison Across Different Environments')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(45, 85)
    
    # 添加SDAN平均线
    if 'SDAN' in results:
        sdan_avg = np.mean(list(results['SDAN'].values()))
        ax.axhline(y=sdan_avg, color='gray', linestyle='--', 
                   label=f'SDAN Avg: {sdan_avg:.1f}%')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_augmentation_ablation(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    绘制数据增强消融实验图 (论文图9)
    
    Args:
        results: 结果字典 {strategy: accuracy}
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    strategies = list(results.keys())
    accuracies = list(results.values())
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(strategies)))
    
    bars = ax.bar(strategies, accuracies, color=colors)
    
    # 添加数值标签和增量
    baseline = results.get('Baseline (No Aug)', results.get(strategies[0], 0))
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
        
        # 增量标注
        if i > 0 and acc > baseline:
            delta = acc - baseline
            ax.annotate(f'+{delta:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height + 1),
                       xytext=(0, 8),
                       textcoords="offset points",
                       ha='center', va='bottom', color='green', fontsize=8)
    
    ax.set_xlabel('Data Augmentation Strategy')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Contribution of Physical Data Augmentation Strategies')
    
    # 添加基线虚线
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_module_ablation(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    绘制网络模块消融实验图 (论文图11)
    
    Args:
        results: 结果字典 {config: accuracy}
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    configs = list(results.keys())
    accuracies = list(results.values())
    
    # 水平条形图
    colors = ['#e74c3c' if 'ResNet' in c else '#3498db' for c in configs]
    
    y_pos = np.arange(len(configs))
    bars = ax.barh(y_pos, accuracies, color=colors)
    
    # 添加数值标签
    full_sdan_acc = results.get('Full SDAN', max(accuracies))
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        delta = acc - full_sdan_acc
        label = f'{acc:.1f}% ({delta:+.1f})' if delta != 0 else f'{acc:.1f}%'
        ax.annotate(label,
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Ablation Study of SDAN Architecture Components')
    
    # 添加Full SDAN基准线
    ax.axvline(x=full_sdan_acc, color='blue', linestyle='--', alpha=0.5)
    
    ax.set_xlim(65, 82)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_augmentation_vs_denoising(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    """
    绘制数据增强vs小波去噪对比图 (论文图10)
    
    Args:
        results: 结果字典 {model: {method: accuracy}}
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results.keys())
    methods = ['No Preprocessing', 'Wavelet Denoising', 'Physical Data Augmentation']
    
    x = np.arange(len(models))
    width = 0.25
    
    colors = ['#95a5a6', '#f39c12', '#27ae60']
    
    for i, method in enumerate(methods):
        accuracies = [results[model].get(method, 0) for model in models]
        bars = ax.bar(x + i * width, accuracies, width, label=method, color=colors[i])
        
        # 添加增量标注
        if i == 2:  # 物理数据增强
            for j, (bar, acc) in enumerate(zip(bars, accuracies)):
                height = bar.get_height()
                baseline = results[models[j]].get('No Preprocessing', 0)
                delta = acc - baseline
                ax.annotate(f'+{delta:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', color='green', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Physical Data Augmentation vs. Wavelet Denoising')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(45, 85)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    绘制训练曲线 (论文图13)
    
    Args:
        history: 训练历史 {metric: [values]}
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = history.get('epochs', range(len(history.get('train_loss', []))))
    
    # 准确率曲线
    ax1 = axes[0]
    if 'train_acc' in history:
        ax1.plot(epochs, history['train_acc'], label='Train', color='blue')
    if 'val_acc' in history:
        ax1.plot(epochs, history['val_acc'], label='Validation', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    ax2 = axes[1]
    if 'train_loss' in history:
        ax2.plot(epochs, history['train_loss'], label='Train', color='blue')
    if 'val_loss' in history:
        ax2.plot(epochs, history['val_loss'], label='Validation', color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_module_contribution_heatmap(
    contributions: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    绘制模块贡献热力图 (论文图12)
    
    Args:
        contributions: 贡献字典 {module: {environment: contribution}}
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    modules = list(contributions.keys())
    environments = list(contributions[modules[0]].keys())
    
    data = np.array([[contributions[m][e] for e in environments] for m in modules])
    
    sns.heatmap(
        data, annot=True, fmt='.1f', cmap='Reds',
        xticklabels=environments, yticklabels=modules,
        ax=ax, cbar_kws={'label': 'Contribution (%)'}
    )
    
    ax.set_xlabel('Environment')
    ax.set_ylabel('Module')
    ax.set_title('Module Contribution Across Environments (%)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def generate_all_figures(
    results_dir: str = "results",
    output_dir: str = "visualize",
):
    """
    生成所有论文图表
    
    Args:
        results_dir: 结果目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例数据（实际使用时从results_dir加载）
    
    # 图7: 性能对比
    performance_results = {
        'AlexNet': {'Empty Room': 71.0, 'Classroom': 70.5, 'Meeting Room': 66.5},
        'ResNet-18': {'Empty Room': 71.5, 'Classroom': 70.8, 'Meeting Room': 67.0},
        'ResNet-TS': {'Empty Room': 70.5, 'Classroom': 67.2, 'Meeting Room': 64.5},
        'RF-Net': {'Empty Room': 62.5, 'Classroom': 58.5, 'Meeting Room': 54.0},
        'SDAN (Ours)': {'Empty Room': 79.6, 'Classroom': 78.2, 'Meeting Room': 75.8},
    }
    plot_performance_comparison(
        performance_results,
        save_path=os.path.join(output_dir, 'fig7_performance_comparison.png')
    )
    
    # 图9: 数据增强消融
    augmentation_results = {
        'Baseline (No Aug)': 68.5,
        '+FDA': 72.5,
        '+TDA': 71.8,
        '+MDA': 73.2,
        '+FDA+TDA': 75.3,
        '+FDA+MDA': 76.0,
        '+TDA+MDA': 75.6,
        'Full (All)': 77.9,
    }
    plot_augmentation_ablation(
        augmentation_results,
        save_path=os.path.join(output_dir, 'fig9_augmentation_ablation.png')
    )
    
    # 图11: 网络模块消融
    module_results = {
        'Replace with ResNet-18': 70.3,
        'w/o Dynamic Attention': 76.3,
        'w/o Time-Freq Decoupling': 75.5,
        'w/o Multi-scale Feature': 74.3,
        'Full SDAN': 77.9,
    }
    plot_module_ablation(
        module_results,
        save_path=os.path.join(output_dir, 'fig11_module_ablation.png')
    )
    
    # 图10: 增强vs去噪
    aug_vs_denoise = {
        'SDAN': {
            'No Preprocessing': 73.5,
            'Wavelet Denoising': 75.2,
            'Physical Data Augmentation': 77.9,
        },
        'ResNet-18': {
            'No Preprocessing': 68.5,
            'Wavelet Denoising': 69.8,
            'Physical Data Augmentation': 70.4,
        },
        'AlexNet': {
            'No Preprocessing': 68.0,
            'Wavelet Denoising': 69.1,
            'Physical Data Augmentation': 69.1,
        },
        'ResNet-TS': {
            'No Preprocessing': 65.5,
            'Wavelet Denoising': 66.8,
            'Physical Data Augmentation': 67.8,
        },
        'RF-Net': {
            'No Preprocessing': 54.5,
            'Wavelet Denoising': 57.5,
            'Physical Data Augmentation': 61.1,
        },
    }
    plot_augmentation_vs_denoising(
        aug_vs_denoise,
        save_path=os.path.join(output_dir, 'fig10_aug_vs_denoise.png')
    )
    
    # 图12: 模块贡献热力图
    module_contributions = {
        'Multi-scale Feature': {'Empty Room': 3.8, 'Classroom': 3.7, 'Meeting Room': 3.3},
        'Time-Freq Decoupling': {'Empty Room': 2.7, 'Classroom': 2.4, 'Meeting Room': 2.0},
        'Dynamic Attention': {'Empty Room': 1.8, 'Classroom': 1.7, 'Meeting Room': 1.3},
    }
    plot_module_contribution_heatmap(
        module_contributions,
        save_path=os.path.join(output_dir, 'fig12_module_contribution.png')
    )
    
    print(f"All figures generated in {output_dir}")


if __name__ == "__main__":
    generate_all_figures()
