#!/usr/bin/env python3
"""
HyperNova 严格验证实验

设计原则：
1. 任务必须足够难，能体现优化器的差异
2. 模型必须足够大，能体现 HyperNova 的 scalability 优势
3. 训练必须足够长，能体现收敛速度差异
4. 对比必须公平，相同计算预算下比较
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 添加项目路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class StrictExperimentConfig:
    """严格实验配置"""
    
    def __init__(self):
        # 固定随机种子，确保可复现
        self.seed = 42
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算预算：固定总迭代次数，公平对比
        self.total_iterations = 10000  # 固定迭代次数，不是 epoch
        
        # 数据配置
        self.dataset = 'cifar100'  # CIFAR-100 比 CIFAR-10 难得多
        self.batch_size = 256  # 更大的 batch size
        
        # 模型配置：使用更大的模型
        self.model_name = 'resnet50'  # ResNet-50 而不是 ResNet-18
        self.num_classes = 100
        
        # 优化器配置
        self.optimizers = {
            'adamw': {
                'lr': 1e-3,
                'weight_decay': 5e-4,
                'betas': (0.9, 0.999),
            },
            'sgd': {
                'lr': 1e-1,
                'weight_decay': 5e-4,
                'momentum': 0.9,
            },
            'hypernova': {
                'lr': 1e-3,
                'weight_decay': 5e-4,
                'betas': (0.9, 0.999),
                'rank_ratio': 0.1,
                'use_gnn': True,
            }
        }
        
        # 学习率调度
        self.lr_schedule = 'cosine'  # cosine annealing
        self.warmup_iterations = 500
        
        # 评估频率
        self.eval_every = 500  # 每 500 迭代评估一次
        
        # 早停 patience（基于验证 loss）
        self.patience = 10  # 10 次评估没有改善就停止
        
        # 保存路径
        self.save_dir = './experiments/strict_results'
        os.makedirs(self.save_dir, exist_ok=True)


def get_dataloaders(config: StrictExperimentConfig):
    """获取数据加载器"""
    
    if config.dataset == 'cifar100':
        # CIFAR-100 更强的数据增强
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
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
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        
    elif config.dataset == 'cifar10':
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
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    
    trainloader = DataLoader(
        trainset, batch_size=config.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=100, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    return trainloader, testloader


def get_model(config: StrictExperimentConfig):
    """获取模型"""
    if config.model_name == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(num_classes=config.num_classes)
    elif config.model_name == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(num_classes=config.num_classes)
    elif config.model_name == 'vgg16':
        from torchvision.models import vgg16_bn
        model = vgg16_bn(num_classes=config.num_classes)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model


def get_optimizer(model: nn.Module, opt_name: str, config: StrictExperimentConfig):
    """创建优化器"""
    opt_config = config.optimizers[opt_name]
    
    if opt_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            betas=opt_config['betas'],
            weight_decay=opt_config['weight_decay']
        )
    
    elif opt_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            momentum=opt_config['momentum'],
            weight_decay=opt_config['weight_decay']
        )
    
    elif opt_name == 'hypernova':
        from hypernova_core import HyperNovaOptimizer
        return HyperNovaOptimizer(
            model.parameters(),
            lr=opt_config['lr'],
            betas=opt_config['betas'],
            weight_decay=opt_config['weight_decay'],
            rank_ratio=opt_config['rank_ratio'],
            use_gnn=opt_config['use_gnn']
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_lr_scheduler(optimizer, config: StrictExperimentConfig):
    """学习率调度器"""
    if config.lr_schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.total_iterations, eta_min=1e-6
        )
    elif config.lr_schedule == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.total_iterations // 3, gamma=0.1
        )
    else:
        return None


def warmup_lr_scheduler(optimizer, current_iter: int, warmup_iters: int, base_lr: float):
    """Warmup 学习率"""
    if current_iter < warmup_iters:
        lr = base_lr * (current_iter / warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_step(model: nn.Module, batch: Tuple, optimizer, criterion, device):
    """单步训练"""
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    
    return loss.item(), correct, total


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def run_strict_experiment(opt_name: str, config: StrictExperimentConfig, 
                          trainloader: DataLoader, testloader: DataLoader):
    """
    运行严格对比实验
    
    关键设计：
    1. 固定迭代次数，不是 epoch
    2. 使用验证 loss 作为早停标准
    3. 记录完整的训练曲线
    4. 测量 wall-clock time
    """
    
    print(f"\n{'='*80}")
    print(f"Running strict experiment: {opt_name}")
    print(f"Model: {config.model_name}, Dataset: {config.dataset}")
    print(f"Total iterations: {config.total_iterations}")
    print(f"{'='*80}\n")
    
    # 创建模型
    model = get_model(config).to(config.device)
    
    # 创建优化器
    optimizer = get_optimizer(model, opt_name, config)
    
    # 学习率调度
    scheduler = get_lr_scheduler(optimizer, config)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 记录指标
    results = {
        'optimizer': opt_name,
        'config': config.optimizers[opt_name],
        'iterations': [],
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'learning_rates': [],
        'wall_clock_times': [],
    }
    
    # 训练循环
    model.train()
    iteration = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 重置内存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    train_iter = iter(trainloader)
    
    while iteration < config.total_iterations:
        # 获取下一个 batch，如果用完就重置
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            batch = next(train_iter)
        
        # Warmup
        warmup_lr_scheduler(
            optimizer, iteration, config.warmup_iterations, 
            config.optimizers[opt_name]['lr']
        )
        
        # 训练步骤
        loss, correct, total = train_step(model, batch, optimizer, criterion, config.device)
        
        # 更新学习率
        if scheduler is not None and iteration >= config.warmup_iterations:
            scheduler.step()
        
        iteration += 1
        
        # 定期评估
        if iteration % config.eval_every == 0 or iteration == config.total_iterations:
            elapsed = time.time() - start_time
            
            # 评估
            val_loss, val_acc = evaluate(model, testloader, criterion, config.device)
            model.train()
            
            # 记录
            results['iterations'].append(iteration)
            results['train_losses'].append(loss)
            results['train_accs'].append(100. * correct / total)
            results['val_losses'].append(val_loss)
            results['val_accs'].append(val_acc)
            results['learning_rates'].append(optimizer.param_groups[0]['lr'])
            results['wall_clock_times'].append(elapsed)
            
            print(f"Iter {iteration:5d}/{config.total_iterations} | "
                  f"Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查（基于验证 loss）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          os.path.join(config.save_dir, f'{opt_name}_best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at iteration {iteration}")
                    break
    
    total_time = time.time() - start_time
    
    # 获取峰值内存
    peak_memory = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    # 最终评估
    final_val_loss, final_val_acc = evaluate(model, testloader, criterion, config.device)
    
    # 汇总结果
    results['total_time'] = total_time
    results['peak_memory_mb'] = peak_memory
    results['final_val_loss'] = final_val_loss
    results['final_val_acc'] = final_val_acc
    results['best_val_loss'] = best_val_loss
    results['iterations_completed'] = iteration
    
    # 计算收敛速度指标
    target_acc = 70.0  # CIFAR-100 上 70% 是合理的 target
    iterations_to_target = None
    for i, acc in enumerate(results['val_accs']):
        if acc >= target_acc:
            iterations_to_target = results['iterations'][i]
            break
    results['iterations_to_70'] = iterations_to_target
    
    # 保存结果
    save_path = os.path.join(config.save_dir, f'{opt_name}_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed: {opt_name}")
    print(f"Final Val Acc: {final_val_acc:.2f}%")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Peak Memory: {peak_memory:.2f} MB")
    if iterations_to_target:
        print(f"Iterations to 70%: {iterations_to_target}")
    print(f"Results saved to: {save_path}")
    print(f"{'='*80}\n")
    
    return results


def plot_comparison(results_dict: Dict[str, dict], config: StrictExperimentConfig):
    """绘制对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 验证准确率曲线
    ax = axes[0, 0]
    for opt_name, results in results_dict.items():
        ax.plot(results['iterations'], results['val_accs'], 
                label=opt_name, linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy vs Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 验证 loss 曲线
    ax = axes[0, 1]
    for opt_name, results in results_dict.items():
        ax.plot(results['iterations'], results['val_losses'], 
                label=opt_name, linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Wall-clock time 对比
    ax = axes[1, 0]
    for opt_name, results in results_dict.items():
        ax.plot(results['wall_clock_times'], results['val_accs'], 
                label=opt_name, linewidth=2)
    ax.set_xlabel('Wall-Clock Time (s)')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 最终指标对比（柱状图）
    ax = axes[1, 1]
    opt_names = list(results_dict.keys())
    final_accs = [results_dict[name]['final_val_acc'] for name in opt_names]
    times = [results_dict[name]['total_time'] for name in opt_names]
    
    x = range(len(opt_names))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar([i - width/2 for i in x], final_accs, width, 
                   label='Final Acc (%)', color='steelblue')
    bars2 = ax2.bar([i + width/2 for i in x], times, width, 
                    label='Time (s)', color='coral')
    
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Final Accuracy (%)', color='steelblue')
    ax2.set_ylabel('Time (s)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(opt_names)
    ax.set_title('Final Metrics Comparison')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, 'comparison.png'), dpi=150)
    print(f"Comparison plot saved to: {os.path.join(config.save_dir, 'comparison.png')}")


def print_summary_table(results_dict: Dict[str, dict]):
    """打印汇总表格"""
    
    print("\n" + "="*100)
    print("STRICT EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print(f"{'Optimizer':<15} {'Final Acc':>12} {'Best Loss':>12} {'Time(s)':>12} "
          f"{'Memory(MB)':>12} {'Iter@70%':>12} {'Iters':>10}")
    print("-"*100)
    
    for opt_name, results in results_dict.items():
        iter_to_70 = results.get('iterations_to_70', 'N/A')
        iter_str = f"{iter_to_70}" if iter_to_70 else "N/A"
        
        print(f"{opt_name:<15} "
              f"{results['final_val_acc']:>12.2f} "
              f"{results['best_val_loss']:>12.4f} "
              f"{results['total_time']:>12.1f} "
              f"{results['peak_memory_mb']:>12.1f} "
              f"{iter_str:>12} "
              f"{results['iterations_completed']:>10}")
    
    print("="*100)
    
    # 计算相对改进
    if 'adamw' in results_dict and 'hypernova' in results_dict:
        adamw = results_dict['adamw']
        hypernova = results_dict['hypernova']
        
        acc_improve = hypernova['final_val_acc'] - adamw['final_val_acc']
        time_ratio = adamw['total_time'] / hypernova['total_time'] if hypernova['total_time'] > 0 else 0
        
        print(f"\nHyperNova vs AdamW:")
        print(f"  Accuracy improvement: {acc_improve:+.2f}%")
        print(f"  Speedup: {time_ratio:.2f}x")
        
        if adamw.get('iterations_to_70') and hypernova.get('iterations_to_70'):
            conv_speedup = adamw['iterations_to_70'] / hypernova['iterations_to_70']
            print(f"  Convergence speedup: {conv_speedup:.2f}x")
    print()


def main():
    """主函数"""
    
    # 创建配置
    config = StrictExperimentConfig()
    
    # 获取数据
    trainloader, testloader = get_dataloaders(config)
    
    print(f"\n{'='*80}")
    print("STRICT HYPERNOVA VALIDATION EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model_name}")
    print(f"Total iterations: {config.total_iterations}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print(f"{'='*80}\n")
    
    # 运行所有优化器对比
    results_dict = {}
    
    for opt_name in ['adamw', 'sgd', 'hypernova']:
        results = run_strict_experiment(opt_name, config, trainloader, testloader)
        results_dict[opt_name] = results
    
    # 打印汇总
    print_summary_table(results_dict)
    
    # 绘制对比图
    try:
        plot_comparison(results_dict, config)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # 保存完整结果
    full_results = {
        'config': {
            'dataset': config.dataset,
            'model': config.model_name,
            'total_iterations': config.total_iterations,
            'batch_size': config.batch_size,
            'seed': config.seed,
        },
        'results': results_dict
    }
    
    with open(os.path.join(config.save_dir, 'full_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nFull results saved to: {os.path.join(config.save_dir, 'full_results.json')}")


if __name__ == "__main__":
    main()
