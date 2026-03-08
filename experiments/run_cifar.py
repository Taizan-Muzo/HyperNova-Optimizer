#!/usr/bin/env python3
"""
CIFAR-10/100 完整对比实验
包含所有基线优化器：AdamW, SGD, Muon, AdaMuon, Sophia, HyperNova
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


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """获取 CIFAR-10 数据加载器"""
    # 数据增强
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
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader


def get_cifar100_loaders(batch_size=128, num_workers=4):
    """获取 CIFAR-100 数据加载器"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader


def get_model(model_name='resnet18', num_classes=10):
    """获取模型"""
    if model_name == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(num_classes=num_classes)
    elif model_name == 'vgg16':
        from torchvision.models import vgg16_bn
        model = vgg16_bn(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def run_cifar10_experiments():
    """运行 CIFAR-10 所有对比实验"""
    print("="*80)
    print("CIFAR-10 对比实验")
    print("="*80)
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # 实验配置
    experiments = [
        # AdamW 基线
        {
            'name': 'adamw',
            'config': ExperimentConfig(
                optimizer_name='adamw',
                lr=1e-3,
                weight_decay=5e-4,
                betas=(0.9, 0.999),
                epochs=200,
                batch_size=128,
                save_dir='./experiments/cifar10/results'
            )
        },
        # SGD with momentum
        {
            'name': 'sgd',
            'config': ExperimentConfig(
                optimizer_name='sgd',
                lr=1e-1,  # SGD needs higher LR
                weight_decay=5e-4,
                epochs=200,
                batch_size=128,
                save_dir='./experiments/cifar10/results'
            )
        },
        # HyperNova - 完整版
        {
            'name': 'hypernova_full',
            'config': ExperimentConfig(
                optimizer_name='hypernova',
                lr=1e-3,
                weight_decay=5e-4,
                betas=(0.9, 0.999),
                rank_ratio=0.1,
                use_gnn=True,
                epochs=200,
                batch_size=128,
                save_dir='./experiments/cifar10/results'
            )
        },
        # HyperNova - 无 GNN
        {
            'name': 'hypernova_no_gnn',
            'config': ExperimentConfig(
                optimizer_name='hypernova',
                lr=1e-3,
                weight_decay=5e-4,
                betas=(0.9, 0.999),
                rank_ratio=0.1,
                use_gnn=False,
                epochs=200,
                batch_size=128,
                save_dir='./experiments/cifar10/results'
            )
        },
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"运行实验: {exp['name']}")
        print(f"{'='*80}")
        
        model = get_model('resnet18', num_classes=10)
        runner = ExperimentRunner(exp['config'])
        result = runner.run_experiment(model, train_loader, test_loader)
        results[exp['name']] = result
    
    # 保存汇总结果
    summary = {}
    for name, result in results.items():
        summary[name] = {
            'best_val_acc': result.best_val_acc,
            'final_val_acc': result.final_val_acc,
            'total_time': result.total_time,
            'peak_memory_mb': result.peak_memory_mb,
            'epochs_to_90': result.epochs_to_90,
            'epochs_to_95': result.epochs_to_95,
        }
    
    with open('./experiments/cifar10/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印对比表
    print("\n" + "="*80)
    print("CIFAR-10 实验结果汇总")
    print("="*80)
    print(f"{'Optimizer':<20} {'Best Acc':>10} {'Time(s)':>12} {'Memory(MB)':>12}")
    print("-"*80)
    for name, s in summary.items():
        print(f"{name:<20} {s['best_val_acc']:>10.2f} {s['total_time']:>12.1f} {s['peak_memory_mb']:>12.1f}")
    print("="*80)
    
    return results


def run_cifar100_experiments():
    """运行 CIFAR-100 对比实验"""
    print("="*80)
    print("CIFAR-100 对比实验")
    print("="*80)
    
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)
    
    experiments = [
        {
            'name': 'adamw',
            'config': ExperimentConfig(
                optimizer_name='adamw',
                lr=1e-3,
                weight_decay=5e-4,
                epochs=200,
                batch_size=128,
                save_dir='./experiments/cifar100/results'
            )
        },
        {
            'name': 'hypernova',
            'config': ExperimentConfig(
                optimizer_name='hypernova',
                lr=1e-3,
                weight_decay=5e-4,
                rank_ratio=0.1,
                use_gnn=True,
                epochs=200,
                batch_size=128,
                save_dir='./experiments/cifar100/results'
            )
        },
    ]
    
    results = {}
    for exp in experiments:
        print(f"\n运行实验: {exp['name']}")
        model = get_model('resnet18', num_classes=100)
        runner = ExperimentRunner(exp['config'])
        result = runner.run_experiment(model, train_loader, test_loader)
        results[exp['name']] = result
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'both'])
    args = parser.parse_args()
    
    if args.dataset in ['cifar10', 'both']:
        run_cifar10_experiments()
    
    if args.dataset in ['cifar100', 'both']:
        run_cifar100_experiments()
