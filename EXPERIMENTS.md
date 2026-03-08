# HyperNova 顶会实验运行指南

## 项目结构

```
optimizers/
├── hypernova/              # 核心库代码
│   ├── config/
│   ├── optimizer/
│   ├── engine/
│   └── tests/
├── experiments/            # 实验脚本
│   ├── experiment_framework.py   # 实验框架
│   ├── run_cifar.py              # CIFAR-10/100 实验
│   ├── run_llama.py              # LLaMA 实验
│   ├── run_ablation.py           # 消融实验
│   └── visualize.py              # 可视化
├── paper/                  # 论文
│   ├── main.tex            # 主论文
│   └── supplementary.tex   # 补充材料
└── hypernova_core.py       # HyperNova 核心实现
```

## 实验清单

### Phase 1: CIFAR-10/100 小规模验证

```bash
# 运行 CIFAR-10 对比实验
python experiments/run_cifar.py --dataset cifar10

# 运行 CIFAR-100 对比实验
python experiments/run_cifar.py --dataset cifar100

# 或者同时运行
python experiments/run_cifar.py --dataset both
```

**预期结果**:
- [ ] AdamW: ~93.4% accuracy
- [ ] SGD: ~93.8% accuracy
- [ ] HyperNova: ~95.2% accuracy
- [ ] HyperNova 内存减少 50%
- [ ] HyperNova 速度提升 1.5x

### Phase 2: ImageNet 大规模验证

```bash
# 需要修改 run_cifar.py 使用 ImageNet 数据集
# 或使用专门的 ImageNet 脚本
```

**预期结果**:
- [ ] AdamW: 77.6% Top-1
- [ ] HyperNova: 78.5% Top-1

### Phase 3: LLaMA 语言模型

```bash
# 运行 LLaMA 对比实验
python experiments/run_llama.py
```

**预期结果**:
- [ ] AdamW: baseline
- [ ] HyperNova: 1.5x 速度提升
- [ ] HyperNova: 50% 内存减少

### Phase 4: 消融实验

```bash
# 运行消融实验
python experiments/run_ablation.py
```

**验证组件**:
- [ ] LSM (对数谱流形)
- [ ] GNN 修正
- [ ] 辛格积分器

### Phase 5: 可视化

```bash
# 生成所有图表
python experiments/visualize.py
```

**输出**:
- [ ] training_curves.png
- [ ] comparison_bar.png
- [ ] convergence_speed.png
- [ ] ablation_study.png

## 环境要求

### 基础环境

```bash
# Python 3.8+
python --version

# PyTorch 1.10+
python -c "import torch; print(torch.__version__)"

# CUDA (如果使用 GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

### 安装依赖

```bash
pip install torch torchvision
pip install transformers  # 用于 LLaMA 实验
pip install matplotlib numpy  # 用于可视化
pip install pytest  # 用于测试
```

### 可选依赖

```bash
pip install wandb  # 实验跟踪
pip install tensorboard  # 可视化
```

## 运行顺序

### 快速验证 (1-2 小时)

```bash
# 1. 运行单元测试
python -m pytest hypernova/tests/test_optimizer.py -v

# 2. 运行小规模 CIFAR-10 (减少 epoch)
# 修改 run_cifar.py 中的 epochs=10
python experiments/run_cifar.py --dataset cifar10

# 3. 生成可视化
python experiments/visualize.py
```

### 完整实验 (1-2 天)

```bash
# 1. CIFAR-10/100 (200 epochs each)
python experiments/run_cifar.py --dataset both

# 2. 消融实验 (100 epochs)
python experiments/run_ablation.py

# 3. LLaMA 实验
python experiments/run_llama.py

# 4. 生成所有图表
python experiments/visualize.py
```

## 结果分析

### 关键指标

1. **准确率**: Best Validation Accuracy
2. **收敛速度**: Epochs to 90%/95%
3. **训练时间**: Total Training Time
4. **内存占用**: Peak Memory Usage

### 对比表格

实验完成后，检查以下表格是否生成:

- `experiments/cifar10/summary.json`
- `experiments/cifar100/summary.json`
- `experiments/ablation/summary.json`
- `experiments/llama/results/summary.json`

### 论文图表

确保以下图表已生成:

- `paper/figures/cifar10_training_curves.png`
- `paper/figures/cifar10_comparison.png`
- `paper/figures/cifar10_convergence.png`
- `paper/figures/ablation_study.png`

## 论文提交准备

### 1. 编译 LaTeX

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 补充材料
pdflatex supplementary.tex
```

### 2. 检查清单

- [ ] 所有实验完成
- [ ] 所有图表生成
- [ ] LaTeX 编译成功
- [ ] 补充材料完整
- [ ] 代码可复现
- [ ] README 更新

### 3. 匿名化

提交前确保:
- [ ] 作者信息匿名化
- [ ] GitHub 链接匿名化
- [ ] 无个人身份信息

## 常见问题

### Q: CUDA out of memory

**解决方案**:
- 减小 batch_size
- 减小 rank_ratio (HyperNova)
- 使用梯度累积

### Q: 实验运行太慢

**解决方案**:
- 减少 epochs (用于快速验证)
- 使用更小的模型
- 使用多 GPU

### Q: 结果与预期不符

**检查项**:
- [ ] 学习率是否正确
- [ ] 数据预处理是否正确
- [ ] 随机种子是否设置
- [ ] 模型初始化是否一致

## 联系与支持

如有问题，请:
1. 检查本指南
2. 查看代码注释
3. 查阅论文补充材料

---

**最后更新**: 2026-03-08
