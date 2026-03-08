# HyperNova Optimizer - 技术文档

## 目录

1. [架构概览](#架构概览)
2. [核心组件](#核心组件)
3. [使用指南](#使用指南)
4. [API 参考](#api-参考)
5. [最佳实践](#最佳实践)

---

## 架构概览

HyperNova 采用分层架构设计：

```
┌─────────────────────────────────────────┐
│         Training Engine (trainer.py)     │
│  - AMP Support                            │
│  - Gradient Accumulation                  │
│  - Checkpointing                          │
│  - Distributed Training Ready             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Parameter Group Manager             │
│  - Layer Type Detection                   │
│  - Strategy Application                   │
│  - Custom Layer Support                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Optimizer Registry               │
│  - AdamW, SGD, Lion, etc.                │
│  - Custom Optimizer Support               │
│  - Pluggable Architecture                 │
└─────────────────────────────────────────┘
```

## 核心组件

### 1. Configuration (`config/schema.py`)

使用 Pydantic 进行类型安全的配置管理：

```python
from hypernova import TrainingEngineConfig, OptimizerConfig

config = TrainingEngineConfig(
    optimizer_cfg=OptimizerConfig(
        name="adamw",
        lr=1e-4,
        weight_decay=0.01,
    ),
    amp=True,
    grad_accum_steps=4,
)
```

### 2. Layer Parameter Group Manager

自动检测层类型并应用优化策略：

```python
from hypernova import LayerOptimizationStrategy

strategies = [
    # 归一化层不使用权重衰减
    LayerOptimizationStrategy(class_name="LayerNorm", weight_decay_scale=0.0),
    
    # Embedding 层使用更高学习率
    LayerOptimizationStrategy(class_name="Embedding", lr_scale=10.0),
    
    # Attention 层使用不同的 betas
    LayerOptimizationStrategy(class_name="MultiheadAttention", betas=(0.9, 0.98)),
]
```

### 3. Production Trainer

生产级训练引擎：

```python
from hypernova import ProductionTrainer

trainer = ProductionTrainer(model, config)

# 训练循环
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader, loss_fn)
    val_loss = trainer.validate(val_loader, loss_fn)
    trainer.save_checkpoint()
```

## 使用指南

### 快速开始

```python
import torch
import torch.nn as nn
from hypernova import ProductionTrainer, TrainingEngineConfig

# 创建模型
model = nn.Transformer(d_model=512, nhead=8)

# 配置
config = TrainingEngineConfig(
    optimizer_cfg=OptimizerConfig(name="adamw", lr=1e-4),
    amp=True,
    grad_accum_steps=4,
)

# 训练器
trainer = ProductionTrainer(model, config)

# 训练
for epoch in range(10):
    avg_loss = trainer.train_epoch(train_loader, loss_fn)
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
```

### 支持的所有层类型

| 层类型 | 示例 | 推荐策略 |
|-------|------|---------|
| Linear | `nn.Linear` | 默认 |
| Conv1D/2D/3D | `nn.Conv2d` | 默认 |
| LayerNorm | `nn.LayerNorm` | weight_decay=0 |
| BatchNorm | `nn.BatchNorm2d` | weight_decay=0 |
| GroupNorm | `nn.GroupNorm` | weight_decay=0 |
| Embedding | `nn.Embedding` | lr_scale=5-10 |
| MultiheadAttention | `nn.MultiheadAttention` | 自定义 betas |

### 模型规模支持

| 规模 | 参数量 | 推荐配置 |
|-----|--------|---------|
| Tiny | < 1M | 单卡，无 AMP |
| Small | 1M-100M | 单卡，AMP |
| Medium | 100M-1B | 多卡 DDP，AMP |
| Large | 1B-10B | FSDP，AMP，grad_accum |
| Massive | >10B | DeepSpeed ZeRO |

## API 参考

### TrainingEngineConfig

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| optimizer_cfg | OptimizerConfig | - | 优化器配置 |
| layer_strategy | List[LayerOptimizationStrategy] | [] | 层策略列表 |
| amp | bool | True | 混合精度 |
| grad_accum_steps | int | 1 | 梯度累积步数 |
| max_grad_norm | float | None | 梯度裁剪 |
| checkpoint_dir | str | "./checkpoints" | 检查点目录 |

### LayerOptimizationStrategy

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| class_name | str | - | 层类名 |
| lr_scale | float | 1.0 | 学习率缩放 |
| weight_decay_scale | float | 1.0 | 权重衰减缩放 |
| betas | tuple | None | 自定义 betas |

## 最佳实践

### 1. Transformer 模型

```python
strategies = [
    LayerOptimizationStrategy(class_name="LayerNorm", weight_decay_scale=0.0),
    LayerOptimizationStrategy(class_name="Embedding", lr_scale=10.0),
    LayerOptimizationStrategy(class_name="MultiheadAttention", betas=(0.9, 0.98)),
]
```

### 2. CNN 模型

```python
strategies = [
    LayerOptimizationStrategy(class_name="BatchNorm", weight_decay_scale=0.0),
    LayerOptimizationStrategy(class_name="Conv2d", lr_scale=1.0),
]
```

### 3. 大模型训练

```python
config = TrainingEngineConfig(
    optimizer_cfg=OptimizerConfig(name="adamw", lr=1e-5),
    amp=True,
    grad_accum_steps=16,  # 大梯度累积
    max_grad_norm=1.0,
    use_spectral_state=True,  # 启用谱状态压缩
    rank_ratio=0.1,  # 10% 秩压缩
)
```

---

**版本**: 0.1.0  
**更新日期**: 2026-03-08
