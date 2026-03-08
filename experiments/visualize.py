#!/usr/bin/env python3
"""
实验结果可视化
生成论文所需的图表
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_experiment_results(experiment_dir):
    """加载实验结果"""
    results = []
    json_files = glob.glob(os.path.join(experiment_dir, 'results', '*.json'))
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def plot_training_curves(results, save_path='./paper/figures/training_curves.png'):
    """绘制训练曲线对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'adamw': '#1f77b4',
        'sgd': '#ff7f0e',
        'hypernova': '#2ca02c',
        'hypernova_full': '#2ca02c',
        'hypernova_no_gnn': '#d62728',
    }
    
    for result in results:
        opt_name = result.get('config', {}).get('optimizer_name', 'unknown')
        color = colors.get(opt_name, '#333333')
        
        # 训练损失
        if 'train_losses' in result:
            axes[0, 0].plot(result['train_losses'], label=opt_name, color=color, alpha=0.8)
        
        # 验证损失
        if 'val_losses' in result:
            axes[0, 1].plot(result['val_losses'], label=opt_name, color=color, alpha=0.8)
        
        # 训练准确率
        if 'train_accuracies' in result:
            axes[1, 0].plot(result['train_accuracies'], label=opt_name, color=color, alpha=0.8)
        
        # 验证准确率
        if 'val_accuracies' in result:
            axes[1, 1].plot(result['val_accuracies'], label=opt_name, color=color, alpha=0.8)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_comparison_bar(summary, save_path='./paper/figures/comparison_bar.png'):
    """绘制对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    optimizers = list(summary.keys())
    best_accs = [summary[opt]['best_val_acc'] for opt in optimizers]
    times = [summary[opt]['total_time'] for opt in optimizers]
    memories = [summary[opt]['peak_memory_mb'] for opt in optimizers]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 准确率对比
    axes[0].bar(optimizers, best_accs, color=colors[:len(optimizers)])
    axes[0].set_title('Best Validation Accuracy')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim([min(best_accs) - 1, max(best_accs) + 1])
    axes[0].tick_params(axis='x', rotation=45)
    
    # 时间对比
    axes[1].bar(optimizers, times, color=colors[:len(optimizers)])
    axes[1].set_title('Total Training Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 内存对比
    axes[2].bar(optimizers, memories, color=colors[:len(optimizers)])
    axes[2].set_title('Peak Memory Usage')
    axes[2].set_ylabel('Memory (MB)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison bar chart saved to: {save_path}")
    plt.close()


def plot_convergence_speed(summary, save_path='./paper/figures/convergence_speed.png'):
    """绘制收敛速度对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizers = list(summary.keys())
    epochs_to_90 = [summary[opt].get('epochs_to_90', 200) for opt in optimizers]
    epochs_to_95 = [summary[opt].get('epochs_to_95', 200) for opt in optimizers]
    
    x = np.arange(len(optimizers))
    width = 0.35
    
    ax.bar(x - width/2, epochs_to_90, width, label='Epochs to 90%', color='#1f77b4')
    ax.bar(x + width/2, epochs_to_95, width, label='Epochs to 95%', color='#ff7f0e')
    
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Epochs')
    ax.set_title('Convergence Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence speed chart saved to: {save_path}")
    plt.close()


def plot_ablation_study(summary, save_path='./paper/figures/ablation_study.png'):
    """绘制消融实验结果"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = list(summary.keys())
    accs = [summary[cfg]['best_val_acc'] for cfg in configs]
    
    colors = ['#2ca02c' if 'full' in cfg else '#1f77b4' if 'adamw' in cfg else '#ff7f0e' for cfg in configs]
    
    bars = ax.barh(configs, accs, color=colors)
    
    # 添加数值标签
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', va='center')
    
    ax.set_xlabel('Validation Accuracy (%)')
    ax.set_title('Ablation Study Results')
    ax.set_xlim([min(accs) - 2, max(accs) + 2])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ablation study chart saved to: {save_path}")
    plt.close()


def generate_all_plots():
    """生成所有图表"""
    print("="*80)
    print("生成实验可视化图表")
    print("="*80)
    
    # 创建输出目录
    os.makedirs('./paper/figures', exist_ok=True)
    
    # CIFAR-10 结果
    if os.path.exists('./experiments/cifar10/results'):
        print("\n处理 CIFAR-10 结果...")
        results = load_experiment_results('./experiments/cifar10')
        if results:
            plot_training_curves(results, './paper/figures/cifar10_training_curves.png')
        
        if os.path.exists('./experiments/cifar10/summary.json'):
            with open('./experiments/cifar10/summary.json', 'r') as f:
                summary = json.load(f)
            plot_comparison_bar(summary, './paper/figures/cifar10_comparison.png')
            plot_convergence_speed(summary, './paper/figures/cifar10_convergence.png')
    
    # 消融实验结果
    if os.path.exists('./experiments/ablation/summary.json'):
        print("\n处理消融实验结果...")
        with open('./experiments/ablation/summary.json', 'r') as f:
            summary = json.load(f)
        plot_ablation_study(summary, './paper/figures/ablation_study.png')
    
    print("\n" + "="*80)
    print("所有图表生成完成！")
    print("图表保存在: ./paper/figures/")
    print("="*80)


if __name__ == "__main__":
    generate_all_plots()
