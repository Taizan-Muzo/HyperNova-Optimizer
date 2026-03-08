"""
AdaMuon 优化器测试脚本
对比 AdamW、Muon 和 AdaMuon 的性能
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('/Users/Zhuanz/.openclaw/workspace/optimizers')
from adamuon import AdaMuon


class SimpleModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_synthetic_data(n_samples=10000, input_dim=784, num_classes=10):
    """创建合成数据集"""
    X = torch.randn(n_samples, input_dim)
    # 添加一些结构使问题更有趣
    true_weights = torch.randn(input_dim, num_classes)
    logits = X @ true_weights
    y = logits.argmax(dim=1)
    return TensorDataset(X, y)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(dataloader), correct / total


def test_optimizer(optimizer_class, optimizer_name, lr=1e-3, epochs=20):
    """测试单个优化器"""
    print(f"\n{'='*50}")
    print(f"Testing {optimizer_name}")
    print('='*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建模型
    model = SimpleModel().to(device)
    
    # 创建优化器
    if optimizer_name == "AdamW":
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name == "AdaMuon":
        optimizer = optimizer_class(model.parameters(), lr=lr, noise_scale=0.001, 
                                   hessian_approx=True, adaptive_ns=True)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    # 创建数据
    train_dataset = create_synthetic_data(n_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # 训练
    losses = []
    accuracies = []
    times = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        loss, acc = train_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - epoch_start
        
        losses.append(loss)
        accuracies.append(acc)
        times.append(epoch_time)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Loss={loss:.4f}, Acc={acc:.4f}, Time={epoch_time:.3f}s")
    
    total_time = time.time() - start_time
    final_acc = accuracies[-1]
    
    print(f"\nFinal Results:")
    print(f"  Final Accuracy: {final_acc:.4f}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Epoch Time: {sum(times)/len(times):.3f}s")
    
    return {
        'name': optimizer_name,
        'losses': losses,
        'accuracies': accuracies,
        'times': times,
        'final_acc': final_acc,
        'total_time': total_time
    }


def plot_results(results_list):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss曲线
    ax = axes[0]
    for result in results_list:
        ax.plot(result['losses'], label=result['name'], marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax = axes[1]
    for result in results_list:
        ax.plot(result['accuracies'], label=result['name'], marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 对比柱状图
    ax = axes[2]
    names = [r['name'] for r in results_list]
    final_accs = [r['final_acc'] for r in results_list]
    total_times = [r['total_time'] for r in results_list]
    
    x = range(len(names))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], final_accs, width, label='Final Accuracy', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar([i + width/2 for i in x], total_times, width, label='Total Time', alpha=0.8, color='orange')
    
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Time (s)')
    ax.set_title('Final Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/Users/Zhuanz/.openclaw/workspace/optimizers/test_results.png', dpi=150)
    print(f"\n结果图已保存: test_results.png")


def main():
    """主测试函数"""
    print("="*60)
    print("AdaMuon Optimizer Benchmark")
    print("="*60)
    
    # 测试配置
    LR = 1e-3
    EPOCHS = 20
    
    results = []
    
    # 测试 AdamW
    try:
        result = test_optimizer(torch.optim.AdamW, "AdamW", lr=LR, epochs=EPOCHS)
        results.append(result)
    except Exception as e:
        print(f"AdamW test failed: {e}")
    
    # 测试 AdaMuon
    try:
        result = test_optimizer(AdaMuon, "AdaMuon", lr=LR, epochs=EPOCHS)
        results.append(result)
    except Exception as e:
        print(f"AdaMuon test failed: {e}")
    
    # 绘制结果
    if len(results) > 0:
        plot_results(results)
        
        # 打印对比
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        for r in results:
            print(f"{r['name']:10s}: Final Acc={r['final_acc']:.4f}, Time={r['total_time']:.2f}s")
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
