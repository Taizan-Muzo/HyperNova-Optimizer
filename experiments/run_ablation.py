#!/usr/bin/env python3
"""
消融实验
验证 HyperNova 各组件的贡献
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.experiment_framework import ExperimentRunner, ExperimentConfig
import json


def get_cifar10_loaders(batch_size=128):
    """获取 CIFAR-10 数据"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def get_model():
    """获取 ResNet-18 模型"""
    from torchvision.models import resnet18
    return resnet18(num_classes=10)


def run_ablation_experiments():
    """运行消融实验"""
    print("="*80)
    print("HyperNova 消融实验")
    print("="*80)
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # 消融实验配置
    experiments = [
        {
            'name': 'full_hypernova',
            'description': '完整的 HyperNova (LSM + Symplectic + GNN)',
            'config': ExperimentConfig(
                optimizer_name='hypernova',
                lr=1e-3,
                weight_decay=5e-4,
                rank_ratio=0.1,
                use_gnn=True,
                epochs=100,  # 消融实验用较少epoch
                batch_size=128,
                save_dir='./experiments/ablation/results'
            )
        },
        {
            'name': 'no_lsm_fp32',
            'description': '无 LSM，使用 FP32 存储',
            'config': ExperimentConfig(
                optimizer_name='hypernova',
                lr=1e-3,
                weight_decay=5e-4,
                rank_ratio=1.0,  # 禁用压缩
                use_gnn=True,
                epochs=100,
                batch_size=128,
                save_dir='./experiments/ablation/results'
            )
        },
        {
            'name': 'no_gnn',
            'description': '无 GNN 修正',
            'config': ExperimentConfig(
                optimizer_name='hypernova',
                lr=1e-3,
                weight_decay=5e-4,
                rank_ratio=0.1,
                use_gnn=False,  # 禁用 GNN
                epochs=100,
                batch_size=128,
                save_dir='./experiments/ablation/results'
            )
        },
        {
            'name': 'no_symplectic',
            'description': '无辛格积分 (使用标准动量)',
            'config': ExperimentConfig(
                optimizer_name='adamw',  # 使用 AdamW 作为无 symplectic 的代理
                lr=1e-3,
                weight_decay=5e-4,
                epochs=100,
                batch_size=128,
                save_dir='./experiments/ablation/results'
            )
        },
        {
            'name': 'adamw_baseline',
            'description': 'AdamW 基线',
            'config': ExperimentConfig(
                optimizer_name='adamw',
                lr=1e-3,
                weight_decay=5e-4,
                epochs=100,
                batch_size=128,
                save_dir='./experiments/ablation/results'
            )
        },
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"实验: {exp['name']}")
        print(f"描述: {exp['description']}")
        print(f"{'='*80}")
        
        model = get_model()
        runner = ExperimentRunner(exp['config'])
        result = runner.run_experiment(model, train_loader, test_loader)
        results[exp['name']] = {
            'description': exp['description'],
            'result': result
        }
    
    # 保存结果
    os.makedirs('./experiments/ablation/results', exist_ok=True)
    summary = {}
    for name, data in results.items():
        summary[name] = {
            'description': data['description'],
            'best_val_acc': data['result'].best_val_acc,
            'final_val_acc': data['result'].final_val_acc,
            'total_time': data['result'].total_time,
            'peak_memory_mb': data['result'].peak_memory_mb,
        }
    
    with open('./experiments/ablation/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印消融表
    print("\n" + "="*80)
    print("消融实验结果汇总")
    print("="*80)
    print(f"{'Configuration':<25} {'Best Acc':>10} {'Time(s)':>10} {'Memory(MB)':>12}")
    print("-"*80)
    for name, s in summary.items():
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<25} {s['best_val_acc']:>10.2f} {s['total_time']:>10.1f} {s['peak_memory_mb']:>12.1f}")
    print("="*80)
    
    # 计算各组件贡献
    print("\n组件贡献分析:")
    print("-"*80)
    baseline_acc = summary['adamw_baseline']['best_val_acc']
    full_acc = summary['full_hypernova']['best_val_acc']
    
    print(f"AdamW 基线: {baseline_acc:.2f}%")
    print(f"完整 HyperNova: {full_acc:.2f}%")
    print(f"总体提升: +{full_acc - baseline_acc:.2f}%")
    print()
    
    # 各组件贡献
    no_lsm_acc = summary['no_lsm_fp32']['best_val_acc']
    no_gnn_acc = summary['no_gnn']['best_val_acc']
    
    print(f"LSM 贡献: +{full_acc - no_lsm_acc:.2f}%")
    print(f"GNN 贡献: +{full_acc - no_gnn_acc:.2f}%")
    
    return results


if __name__ == "__main__":
    run_ablation_experiments()
