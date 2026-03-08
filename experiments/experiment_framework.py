"""
HyperNova 完整实验框架

包含所有顶会需要的对比实验：
1. CIFAR-10/100 小规模验证
2. ImageNet 大规模验证  
3. LLaMA 语言模型训练
4. 消融实验
5. 理论验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Callable
import json
import time
import os
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 优化器设置
    optimizer_name: str = "hypernova"  # hypernova, adamw, muon, adamuon, sophia
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # HyperNova 特有
    rank_ratio: float = 0.1
    use_gnn: bool = True
    
    # 训练设置
    batch_size: int = 128
    epochs: int = 200
    grad_accum_steps: int = 1
    
    # 实验设置
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "./results"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentResult:
    """实验结果"""
    config: ExperimentConfig
    
    # 训练指标
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    
    # 时间指标
    time_per_epoch: List[float]
    total_time: float
    
    # 内存指标
    peak_memory_mb: float
    
    # 最终指标
    final_train_acc: float
    final_val_acc: float
    best_val_acc: float
    
    # 收敛指标
    epochs_to_90: int  # 达到90%准确率需要的epoch
    epochs_to_95: int  # 达到95%准确率需要的epoch
    
    def save(self, path: str):
        """保存结果到JSON"""
        data = {
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'time_per_epoch': self.time_per_epoch,
            'total_time': self.total_time,
            'peak_memory_mb': self.peak_memory_mb,
            'final_train_acc': self.final_train_acc,
            'final_val_acc': self.final_val_acc,
            'best_val_acc': self.best_val_acc,
            'epochs_to_90': self.epochs_to_90,
            'epochs_to_95': self.epochs_to_95,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
    
    def get_optimizer(self, model: nn.Module):
        """根据配置创建优化器"""
        name = self.config.optimizer_name.lower()
        
        if name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        
        elif name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        elif name == "hypernova":
            # 导入 HyperNova
            from hypernova_core import HyperNovaOptimizer
            return HyperNovaOptimizer(
                model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
                rank_ratio=self.config.rank_ratio,
                use_gnn=self.config.use_gnn
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def train_epoch(self, model: nn.Module, loader: DataLoader, 
                    optimizer, criterion) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 梯度累积
            if batch_idx % self.config.grad_accum_steps == 0:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets) / self.config.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                optimizer.step()
            
            total_loss += loss.item() * self.config.grad_accum_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model: nn.Module, loader: DataLoader, 
                 criterion) -> Tuple[float, float]:
        """验证"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def run_experiment(self, model: nn.Module, train_loader: DataLoader, 
                       val_loader: DataLoader) -> ExperimentResult:
        """运行完整实验"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {self.config.optimizer_name}")
        print(f"Config: {self.config.to_dict()}")
        print(f"{'='*60}\n")
        
        model = model.to(self.device)
        optimizer = self.get_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        
        # 记录指标
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        times = []
        
        best_val_acc = 0.0
        epochs_to_90, epochs_to_95 = -1, -1
        
        # 记录内存
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            epoch_time = time.time() - epoch_start
            times.append(epoch_time)
            
            # 记录
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # 记录收敛epoch
            if epochs_to_90 == -1 and val_acc >= 90.0:
                epochs_to_90 = epoch
            if epochs_to_95 == -1 and val_acc >= 95.0:
                epochs_to_95 = epoch
            
            # 打印进度
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                      f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                      f"Time: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # 获取峰值内存
        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        # 创建结果
        result = ExperimentResult(
            config=self.config,
            train_losses=train_losses,
            train_accuracies=train_accs,
            val_losses=val_losses,
            val_accuracies=val_accs,
            time_per_epoch=times,
            total_time=total_time,
            peak_memory_mb=peak_memory,
            final_train_acc=train_accs[-1],
            final_val_acc=val_accs[-1],
            best_val_acc=best_val_acc,
            epochs_to_90=epochs_to_90 if epochs_to_90 != -1 else self.config.epochs,
            epochs_to_95=epochs_to_95 if epochs_to_95 != -1 else self.config.epochs,
        )
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            self.config.save_dir,
            f"{self.config.optimizer_name}_{timestamp}.json"
        )
        result.save(save_path)
        
        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Final Val Acc: {result.final_val_acc:.2f}%")
        print(f"Best Val Acc: {result.best_val_acc:.2f}%")
        print(f"Total Time: {result.total_time:.2f}s")
        print(f"Peak Memory: {result.peak_memory_mb:.2f} MB")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return result


def run_cifar10_experiments():
    """运行 CIFAR-10 对比实验"""
    print("\n" + "="*60)
    print("CIFAR-10 Experiments")
    print("="*60)
    
    # 数据预处理
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
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # 模型: ResNet-18
    from torchvision.models import resnet18
    
    # 对比实验配置
    optimizers = ["adamw", "sgd", "hypernova"]
    results = {}
    
    for opt_name in optimizers:
        config = ExperimentConfig(
            optimizer_name=opt_name,
            lr=1e-3 if opt_name != "sgd" else 1e-1,
            weight_decay=5e-4,
            epochs=200,
            batch_size=128,
            save_dir="./experiments/cifar/results"
        )
        
        model = resnet18(num_classes=10)
        runner = ExperimentRunner(config)
        result = runner.run_experiment(model, trainloader, testloader)
        results[opt_name] = result
    
    return results


if __name__ == "__main__":
    # 运行 CIFAR-10 实验
    results = run_cifar10_experiments()
    
    # 打印对比结果
    print("\n" + "="*60)
    print("Final Comparison")
    print("="*60)
    for name, result in results.items():
        print(f"{name:15s}: Best Acc={result.best_val_acc:.2f}%, "
              f"Time={result.total_time:.1f}s, "
              f"Memory={result.peak_memory_mb:.1f}MB")
