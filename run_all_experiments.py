#!/usr/bin/env python
"""
完整实验运行脚本

复现论文中的所有实验结果
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.preset import preset
from benchmark.experiment_configs import (
    EXPERIMENT_PERFORMANCE_COMPARISON,
    EXPERIMENT_AUGMENTATION_ABLATION,
    EXPERIMENT_MODULE_ABLATION,
    EXPERIMENT_CROSS_ENVIRONMENT,
    EXPECTED_RESULTS,
    FDA_CONFIG, TDA_CONFIG, MDA_CONFIG,
    SDAN_CONFIG, STFT_CONFIG,
)
from benchmark.train import train, train_cross_environment, set_seed
from benchmark.utils import (
    save_results,
    plot_performance_comparison,
    plot_augmentation_ablation,
    plot_module_ablation,
    plot_confusion_matrix,
    generate_all_figures,
)


def run_experiment_1_performance_comparison(config, output_dir):
    """
    实验1: 性能对比 (论文表1, 图7)
    
    比较SDAN与基线模型在不同环境下的性能
    """
    print("\n" + "="*60)
    print("实验1: 性能对比")
    print("="*60)
    
    exp_config = EXPERIMENT_PERFORMANCE_COMPARISON
    all_results = {}
    
    for model_name in exp_config['models']:
        model_results = {}
        
        for env in exp_config['environments']:
            print(f"\n训练 {model_name} on {env}...")
            
            # 多次重复实验
            accuracies = []
            for repeat in range(exp_config['repeat']):
                config['seed'] = 42 + repeat
                results = train(
                    config,
                    model_name=model_name,
                    augmentation_mode=exp_config['augmentation'],
                    environment=[env],
                    save_model=(repeat == 0)  # 只保存第一次
                )
                accuracies.append(results['final_acc'])
            
            model_results[env] = {
                'mean': np.mean(accuracies) * 100,
                'std': np.std(accuracies) * 100,
            }
            print(f"  {env}: {model_results[env]['mean']:.2f}% ± {model_results[env]['std']:.2f}%")
        
        all_results[model_name] = model_results
    
    # 保存结果
    save_path = os.path.join(output_dir, 'exp1_performance_comparison.json')
    save_results(all_results, save_path)
    
    # 生成图表
    plot_data = {m: {e: all_results[m][e]['mean'] for e in exp_config['environments']} 
                 for m in exp_config['models']}
    plot_performance_comparison(
        plot_data,
        save_path=os.path.join(output_dir, 'figures', 'fig7_performance_comparison.png')
    )
    
    return all_results


def run_experiment_2_augmentation_ablation(config, output_dir):
    """
    实验2: 数据增强消融 (论文图9)
    
    评估各数据增强策略的贡献
    """
    print("\n" + "="*60)
    print("实验2: 数据增强消融实验")
    print("="*60)
    
    exp_config = EXPERIMENT_AUGMENTATION_ABLATION
    all_results = {}
    
    for aug_mode, aug_name in exp_config['augmentation_modes']:
        print(f"\n测试增强策略: {aug_name} ({aug_mode})...")
        
        accuracies = []
        for repeat in range(exp_config['repeat']):
            config['seed'] = 42 + repeat
            results = train(
                config,
                model_name=exp_config['model'],
                augmentation_mode=aug_mode,
                save_model=(repeat == 0)
            )
            accuracies.append(results['final_acc'])
        
        all_results[aug_name] = {
            'mean': np.mean(accuracies) * 100,
            'std': np.std(accuracies) * 100,
        }
        print(f"  {aug_name}: {all_results[aug_name]['mean']:.2f}% ± {all_results[aug_name]['std']:.2f}%")
    
    # 保存结果
    save_path = os.path.join(output_dir, 'exp2_augmentation_ablation.json')
    save_results(all_results, save_path)
    
    # 生成图表
    plot_data = {k: v['mean'] for k, v in all_results.items()}
    plot_augmentation_ablation(
        plot_data,
        save_path=os.path.join(output_dir, 'figures', 'fig9_augmentation_ablation.png')
    )
    
    return all_results


def run_experiment_3_module_ablation(config, output_dir):
    """
    实验3: SDAN模块消融 (论文图11)
    
    评估SDAN各模块的贡献
    """
    print("\n" + "="*60)
    print("实验3: SDAN模块消融实验")
    print("="*60)
    
    exp_config = EXPERIMENT_MODULE_ABLATION
    all_results = {}
    
    for model_name, ablation, display_name in exp_config['ablation_configs']:
        print(f"\n测试: {display_name}...")
        
        accuracies = []
        for repeat in range(exp_config['repeat']):
            config['seed'] = 42 + repeat
            
            # 创建模型名称
            full_model_name = f"{model_name}_{ablation}" if ablation else model_name
            
            results = train(
                config,
                model_name=full_model_name,
                augmentation_mode='all',
                save_model=(repeat == 0)
            )
            accuracies.append(results['final_acc'])
        
        all_results[display_name] = {
            'mean': np.mean(accuracies) * 100,
            'std': np.std(accuracies) * 100,
        }
        print(f"  {display_name}: {all_results[display_name]['mean']:.2f}%")
    
    # 保存结果
    save_path = os.path.join(output_dir, 'exp3_module_ablation.json')
    save_results(all_results, save_path)
    
    # 生成图表
    plot_data = {k: v['mean'] for k, v in all_results.items()}
    plot_module_ablation(
        plot_data,
        save_path=os.path.join(output_dir, 'figures', 'fig11_module_ablation.png')
    )
    
    return all_results


def run_experiment_4_cross_environment(config, output_dir):
    """
    实验4: 跨环境泛化 (论文表2)
    
    测试模型在不同环境间的泛化能力
    """
    print("\n" + "="*60)
    print("实验4: 跨环境泛化实验")
    print("="*60)
    
    exp_config = EXPERIMENT_CROSS_ENVIRONMENT
    all_results = {}
    
    for train_env in exp_config['train_environments']:
        all_results[train_env] = {}
        
        for test_env in exp_config['test_environments']:
            print(f"\n训练: {train_env} -> 测试: {test_env}...")
            
            if train_env == test_env:
                # 同环境测试
                results = train(
                    config,
                    model_name=exp_config['model'],
                    augmentation_mode=exp_config['augmentation'],
                    environment=[train_env],
                )
                acc = results['final_acc'] * 100
            else:
                # 跨环境测试
                results = train_cross_environment(
                    config,
                    model_name=exp_config['model'],
                    train_env=train_env,
                    test_env=test_env,
                )
                acc = results['final_acc'] * 100
            
            all_results[train_env][test_env] = acc
            print(f"  准确率: {acc:.2f}%")
    
    # 计算跨环境性能下降
    same_env_accs = [all_results[e][e] for e in exp_config['train_environments']]
    cross_env_accs = []
    for train_env in exp_config['train_environments']:
        for test_env in exp_config['test_environments']:
            if train_env != test_env:
                cross_env_accs.append(all_results[train_env][test_env])
    
    performance_drop = np.mean(same_env_accs) - np.mean(cross_env_accs)
    print(f"\n跨环境性能下降: {performance_drop:.2f}%")
    print(f"(论文报告SDAN下降: {EXPECTED_RESULTS['cross_environment_drop']}%)")
    
    # 保存结果
    save_path = os.path.join(output_dir, 'exp4_cross_environment.json')
    save_results(all_results, save_path)
    
    return all_results


def run_all_experiments(config, output_dir):
    """运行所有实验"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    all_results = {}
    
    # 实验1: 性能对比
    all_results['performance_comparison'] = run_experiment_1_performance_comparison(
        config, output_dir
    )
    
    # 实验2: 数据增强消融
    all_results['augmentation_ablation'] = run_experiment_2_augmentation_ablation(
        config, output_dir
    )
    
    # 实验3: 模块消融
    all_results['module_ablation'] = run_experiment_3_module_ablation(
        config, output_dir
    )
    
    # 实验4: 跨环境泛化
    all_results['cross_environment'] = run_experiment_4_cross_environment(
        config, output_dir
    )
    
    # 保存总结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(
        all_results,
        os.path.join(output_dir, f'all_results_{timestamp}.json')
    )
    
    print("\n" + "="*60)
    print("所有实验完成!")
    print("="*60)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='复现论文实验')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'performance', 'augmentation', 'module', 'cross_env'],
                       help='要运行的实验')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--quick', action='store_true', help='快速测试模式(减少重复次数)')
    args = parser.parse_args()
    
    # 更新配置
    config = preset.copy()
    config['seed'] = args.seed
    
    if args.quick:
        # 快速测试模式
        config['train']['epochs'] = 10
        EXPERIMENT_PERFORMANCE_COMPARISON['repeat'] = 1
        EXPERIMENT_AUGMENTATION_ABLATION['repeat'] = 1
        EXPERIMENT_MODULE_ABLATION['repeat'] = 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.experiment == 'all':
        run_all_experiments(config, args.output_dir)
    elif args.experiment == 'performance':
        run_experiment_1_performance_comparison(config, args.output_dir)
    elif args.experiment == 'augmentation':
        run_experiment_2_augmentation_ablation(config, args.output_dir)
    elif args.experiment == 'module':
        run_experiment_3_module_ablation(config, args.output_dir)
    elif args.experiment == 'cross_env':
        run_experiment_4_cross_environment(config, args.output_dir)


if __name__ == '__main__':
    main()
