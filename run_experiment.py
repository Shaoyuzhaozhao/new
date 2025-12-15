#!/usr/bin/env python
"""
WiFi多用户感知系统 - 主实验运行脚本

使用方法:
    python run_experiment.py --model SDAN --task activity --augmentation all
    python run_experiment.py --model SDAN --augmentation none
    python run_experiment.py --model ResNet18 --task activity
    python run_experiment.py --model SDAN --train_env classroom --test_env meeting_room
    python run_experiment.py --generate_figures
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.preset import preset
from benchmark.train import train, train_cross_environment, set_seed
from benchmark.utils import generate_all_figures, save_results


def parse_args():
    parser = argparse.ArgumentParser(description='WiFi多用户感知系统实验')
    parser.add_argument('--model', type=str, default='SDAN',
                        choices=['SDAN', 'SDAN_no_multiscale', 'SDAN_no_tfdecoupling',
                                 'SDAN_no_attention', 'AlexNet', 'ResNet18', 'ResNet_TS', 'RF_Net'])
    parser.add_argument('--task', type=str, default='activity')
    parser.add_argument('--augmentation', type=str, default='all',
                        choices=['all', 'fda', 'tda', 'mda', 'fda_tda', 'fda_mda', 'tda_mda', 'none'])
    parser.add_argument('--environment', type=str, nargs='+',
                        default=['classroom', 'meeting_room', 'empty_room'])
    parser.add_argument('--train_env', type=str, default=None)
    parser.add_argument('--test_env', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--generate_figures', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def update_config(config, args):
    config = config.copy()
    config['data']['environment'] = args.environment
    config['train']['epochs'] = args.epochs
    config['train']['batch_size'] = args.batch_size
    config['train']['learning_rate'] = args.lr
    config['train']['weight_decay'] = args.weight_decay
    config['train']['dropout'] = args.dropout
    config['seed'] = args.seed
    config['device']['gpu_id'] = args.gpu
    config['output']['path_result'] = os.path.join(args.output_dir, 'result.json')
    config['output']['path_model'] = os.path.join(args.output_dir, 'models')
    config['output']['path_log'] = os.path.join(args.output_dir, 'logs')
    config['output']['path_figure'] = os.path.join(args.output_dir, 'figures')
    return config


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    config = update_config(preset, args)
    set_seed(args.seed)

    if args.generate_figures:
        print("Generating figures...")
        generate_all_figures(args.output_dir, os.path.join(args.output_dir, 'figures'))
        return

    if args.train_env and args.test_env:
        results = train_cross_environment(config, args.model, args.train_env, args.test_env)
    else:
        all_results = []
        for i in range(args.repeat):
            config['seed'] = args.seed + i
            results = train(config, args.model, args.augmentation, args.environment)
            all_results.append(results)

        if args.repeat > 1:
            accs = [r['final_acc'] for r in all_results]
            print(f"\nMean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        results = all_results[0] if args.repeat == 1 else all_results

    save_results(results, os.path.join(args.output_dir, 'result.json'))
    print("Done!")


if __name__ == '__main__':
    main()