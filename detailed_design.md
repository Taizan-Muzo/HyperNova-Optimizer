# AdaMuon 优化器详细设计文档

## 一、设计动机与背景

### 1.1 为什么需要改进 Muon？

Muon 优化器在训练速度上取得了显著成果，但在实际使用中发现以下问题：

**问题1：固定系数的局限性**
- Muon 使用固定的 Newton-Schulz 系数 (3.4445, -4.7750, 2.0315)
- 这些系数是针对特定矩阵规模优化的，不具备普适性
- 在训练早期（梯度大）和后期（梯度小）应该使用不同的收敛策略

**问题2：缺乏自适应性**
- Muon 没有利用 Adam 风格的自适应学习率
- 不同参数层、不同训练阶段需要不同的更新强度
- 缺乏对梯度历史的利用

**问题3：局部最优困境**
- 正交化更新虽然稳定，但可能过早收敛到局部最优
- 缺乏显式的探索机制

**问题4：维度限制**
- Muon 只适用于 2D 参数，对其他维度参数没有统一处理方案

### 1.2 设计目标

设计一个满足以下要求的优化器：
1. ✅ 收敛速度比 Muon 更快
2. ✅ 数值稳定性更好（自适应调整）
3. ✅ 不容易陷入局部最优（噪声注入）
4. ✅ 显存占用不超过 AdamW
5. ✅ 适用于各种维度的参数

---

## 二、核心思想详解

### 2.1 自适应 Newton-Schulz 系数

#### 理论基础

Newton-Schulz 迭代的形式为：
```
X_{k+1} = a*X + b*(X@X.T)@X + c*(X@X.T)^2@X
```

系数 (a,b,c) 决定了：
- **收敛速度**: a 越大，小奇异值收敛越快
- **稳定性**: 需要保证 |φ'(x)| < 1 在 [0,1] 区间内

#### 核心洞察

梯度矩阵 G 的谱范数 ||G||₂ 反映了当前更新的"强度"：
- **训练早期**: 梯度大，谱范数大，需要更快的收敛速度
- **训练后期**: 梯度小，谱范数小，需要更高的稳定性
- **不同层**: 不同层的梯度尺度不同，需要不同的系数

#### 自适应策略

```python
# 估计谱范数（幂迭代，2步足够）
u = randn(m, 1)
for _ in range(2):
    u = G @ (G.T @ u)
    u = u / norm(u)
spectral_norm = sqrt(u.T @ G @ G.T @ u)

# 根据谱范数调整系数
scale = clamp(spectral_norm, 0.5, 2.0)
a, b, c = base_a * scale, base_b * scale, base_c * scale
```

**为什么有效？**
- 谱范数大 → scale > 1 → 系数增大 → 收敛加快
- 谱范数小 → scale < 1 → 系数减小 → 稳定性提高
- 限制在 [0.5, 2.0] 防止极端值

#### 数值稳定性保障

```python
# 检查迭代是否发散
if has_nan(X) or has_inf(X):
    # 回退到简单动量更新
    X = G
```

---

### 2.2 混合更新策略

#### 问题分析

神经网络参数有不同维度：
- **2D 权重矩阵** (如 Linear: [out_dim, in_dim]): 适合正交化
- **1D 偏置向量** (如 bias: [dim]): 正交化无意义
- **3D+ 卷积核** (如 Conv: [out, in, k, k]): 需要特殊处理

#### 统一框架

```python
if param.dim() == 2:
    # 2D: 使用 Muon 正交化更新
    update = orthogonalize(momentum)
else:
    # 1D/3D: 使用 Adam 风格更新
    update = adam_style(momentum, second_moment)
```

#### 2D 参数的正交化更新

**为什么正交化有帮助？**

考虑 SVD 分解：G = UΣV^T
- 正交化后：Ortho(G) = UV^T
- 这相当于去除了奇异值的幅度信息，保留了方向
- 效果：
  1. 均衡不同方向的学习率
  2. 防止某些方向更新过大/过小
  3. 隐式正则化（类似于权重归一化）

**Newton-Schulz vs SVD**

| 方法 | 计算复杂度 | 精度 | 稳定性 |
|-----|----------|-----|-------|
| SVD | O(n³) | 精确 | 高 |
| NS(5步) | O(n²) | 近似 | 中 |

NS 迭代用 5 步达到足够精度，速度快 10-100 倍。

#### 1D/3D 参数的 Adam 更新

对于不适合正交化的参数，使用 Adam 的自适应机制：

```python
# 二阶矩估计
v = beta2 * v + (1 - beta2) * g²
# 自适应学习率
update = m / (sqrt(v) + eps)
```

**优势**：
- 逐元素自适应，适合稀疏梯度
- 计算简单，内存开销小
- 经过广泛验证的稳定性

---

### 2.3 Hessian 近似

#### 动机

一阶方法（SGD, Adam）只利用梯度信息：
```
θ ← θ - lr * g
```

二阶方法（Newton）利用曲率信息：
```
θ ← θ - H^{-1} * g
```

但真实 Hessian H 计算成本太高。

#### 近似策略

使用梯度外积近似 Hessian 对角线：
```
H_ii ≈ E[g_i²]
```

这正是 Adam 的 v 项！但 Adam 是逐元素的，我们希望是结构化的。

#### 实现方式

对于 2D 权重矩阵 W ∈ R^{m×n}：

```python
# 对每一行（每个输出维度）计算梯度范数
grad_norm_per_row = mean(g², dim=1)  # shape: [m]

# 更新 Hessian 对角线近似
hessian_diag = 0.9 * hessian_diag + 0.1 * grad_norm_per_row

# 自适应缩放
adaptive_scale = 1.0 / (sqrt(hessian_diag) + eps)
update = update * adaptive_scale.unsqueeze(1)
```

**效果**：
- 每个输出维度有自己的学习率缩放
- 梯度大的维度（曲率大）学习率减小
- 梯度小的维度（曲率小）学习率增大
- 类似于 AdaGrad，但更轻量

#### 与 Adam 的关系

| 方法 | 缩放方式 | 内存 |
|-----|---------|-----|
| Adam | 逐元素: 1/√v_ij | 2x 参数 |
| Hessian近似 | 逐行: 1/√h_i | 1x 参数 + m维向量 |

Hessian 近似更结构化，内存开销更小。

---

### 2.4 噪声注入

#### 动机

优化过程容易陷入：
- **局部最优**: 梯度为零但不是全局最优
- **鞍点**: 某些方向梯度为零，高维空间中很常见
- **平坦区域**: 梯度很小，收敛缓慢

#### 噪声的作用

```python
update = update + noise_scale * std(update) * randn_like(update)
```

**物理类比**：
- 类似于模拟退火：高温时噪声大，低温时噪声小
- 类似于 SGD 的小批量噪声：帮助逃离尖锐极小值

#### 自适应噪声强度

```python
# 方案1: 固定小噪声
noise_scale = 0.001

# 方案2: 随时间衰减
noise_scale = initial_noise * decay_rate ** step

# 方案3: 基于梯度大小
noise_scale = base_noise / (1 + norm(grad))
```

**推荐**：固定小噪声 (0.001-0.01)，简单有效。

#### 理论依据

噪声注入的益处：
1. **逃离局部最优**: 噪声提供动能越过势垒
2. **平滑损失景观**: 等效于对损失函数做卷积
3. **更好的泛化**: 帮助找到更宽的极小值（flat minima）

---

### 2.5 内存优化

#### 约束

显存占用 ≤ AdamW：
- AdamW: 存储 m, v → 2x 参数
- 目标: ≤ 2x 参数

#### 优化策略

**1. 状态共享**
```python
# 不存储完整的正交化矩阵
# 只在计算时临时生成
update = newton_schulz(momentum)  # 临时计算，不存储
```

**2. 精度优化**
```python
# NS 迭代使用 bfloat16
X = G.bfloat16()
# ... 迭代 ...
X = X.float()  # 转回 float32
```

bfloat16 减少 50% 内存，现代 GPU 支持快速计算。

**3. 选择性存储**
```python
# 2D 参数: 存储 m, hessian_diag (m维)
# 1D/3D 参数: 存储 m, v
```

**内存对比**：

| 优化器 | 存储内容 | 内存 |
|-------|---------|-----|
| SGD | m | 1x |
| AdamW | m, v | 2x |
| Muon | m + 临时 | ~1.5x |
| AdaMuon | m, v/hessian | ~2x |

满足约束！

---

## 三、算法流程

### 3.1 伪代码

```python
class AdaMuon:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, ns_steps=5, noise_scale=0.001):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.noise_scale = noise_scale
    
    def step(self, params, grads):
        for p, g in zip(params, grads):
            # 1. 权重衰减
            p *= (1 - self.lr * self.weight_decay)
            
            # 2. 更新动量
            m = self.state[p]['momentum']
            m = beta1 * m + (1 - beta1) * g
            self.state[p]['momentum'] = m
            
            if p.dim() == 2:
                # ===== 2D 参数 =====
                # 3a. 自适应 NS 系数
                spectral_norm = estimate_spectral_norm(m)
                a, b, c = get_adaptive_coeffs(spectral_norm)
                
                # 4a. Newton-Schulz 正交化
                update = newton_schulz(m, a, b, c, self.ns_steps)
                
                # 5a. Hessian 近似缩放
                h = self.state[p]['hessian_diag']
                h = update_hessian_approx(h, m)
                update *= adaptive_scale(h)
                
            else:
                # ===== 1D/3D 参数 =====
                # 3b. 更新二阶矩
                v = self.state[p]['exp_avg_sq']
                v = beta2 * v + (1 - beta2) * g**2
                self.state[p]['exp_avg_sq'] = v
                
                # 4b. Adam 风格更新
                update = m / (sqrt(v) + self.eps)
            
            # 6. 噪声注入
            if self.noise_scale > 0:
                noise = randn_like(update) * self.noise_scale * std(update)
                update += noise
            
            # 7. 参数更新
            p -= self.lr * update
```

### 3.2 关键子程序

**谱范数估计**（幂迭代）：
```python
def estimate_spectral_norm(G, num_iters=2):
    """估计矩阵的谱范数（最大奇异值）"""
    m, n = G.shape
    u = randn(m, 1)
    
    for _ in range(num_iters):
        u = G @ (G.T @ u)
        u = u / norm(u)
    
    # σ_max ≈ sqrt(u^T G G^T u)
    return sqrt(u.T @ G @ G.T @ u)
```

复杂度: O(num_iters * m * n)，2次迭代足够准确。

**Newton-Schulz 迭代**：
```python
def newton_schulz(G, a, b, c, steps=5):
    """Newton-Schulz 正交化迭代"""
    X = G.bfloat16()
    X = X / (norm(X) + 1e-7)
    
    # 转置优化
    if G.shape[0] > G.shape[1]:
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.shape[0] > G.shape[1]:
        X = X.T
    
    return X.float()
```

---

## 四、理论分析

### 4.1 收敛性

**定理**（非正式）：
在满足以下条件时，AdaMuon 收敛：
1. 损失函数 L 是光滑的（Lipschitz 连续梯度）
2. 学习率满足 Robbins-Monro 条件：Σ lr_t = ∞, Σ lr_t² < ∞
3. 噪声有界

**证明思路**：
- 自适应 NS 系数保证迭代收敛（谱半径 < 1）
- Hessian 近似提供曲率信息，加速收敛
- 噪声注入不影响收敛性（随时间衰减）

### 4.2 与现有方法的关系

**AdaMuon vs Adam**：
- Adam: 对角自适应（逐元素）
- AdaMuon: 全矩阵预处理 + 对角自适应
- AdaMuon 可以看作 Adam 的矩阵推广

**AdaMuon vs Muon**：
- Muon: 固定系数，纯正交化
- AdaMuon: 自适应系数，正交化 + 自适应缩放
- AdaMuon 更灵活，适应性更强

**AdaMuon vs K-FAC**：
- K-FAC: 完整的 Fisher 信息矩阵近似
- AdaMuon: 轻量级的 Hessian 对角线近似
- AdaMuon 计算成本远低于 K-FAC

### 4.3 隐式正则化

正交化更新等价于对权重矩阵施加约束：
```
min L(θ) + λ * ||WW^T - I||²
```

这鼓励权重矩阵保持正交性，有以下好处：
1. 防止权重共线（co-linearity）
2. 保持梯度流的稳定性
3. 类似于 Dropout 的正则化效果

---

## 五、实验设计

### 5.1 验证指标

**主要指标**：
- 收敛速度：达到目标 loss 的 epoch 数
- 最终性能：验证集准确率/loss
- 训练时间：wall-clock time
- 显存占用：peak memory usage

**次要指标**：
- 梯度范数变化
- 学习率自适应情况
- 噪声注入效果

### 5.2 基线对比

**对比方法**：
1. SGD + Momentum
2. AdamW
3. Muon（原始）
4. AdaMuon（改进）

**公平性保证**：
- 相同的学习率调度
- 相同的权重衰减
- 相同的 batch size
- 相同的随机种子

### 5.3 消融实验

测试各组件的贡献：

| 实验 | 自适应NS | Hessian | 噪声 | 混合策略 |
|-----|---------|---------|-----|---------|
| 1 | ✗ | ✗ | ✗ | ✗ (纯Muon) |
| 2 | ✓ | ✗ | ✗ | ✗ |
| 3 | ✓ | ✓ | ✗ | ✗ |
| 4 | ✓ | ✓ | ✓ | ✗ |
| 5 | ✓ | ✓ | ✓ | ✓ (完整AdaMuon) |

### 5.4 超参数敏感性

测试关键超参数：
- noise_scale: [0, 0.001, 0.01, 0.1]
- ns_steps: [3, 5, 7, 10]
- hessian_approx: [True, False]

---

## 六、潜在问题与对策

### 6.1 数值稳定性

**问题**: Newton-Schulz 迭代可能发散

**对策**:
1. 添加 eps 防止除零
2. 检查 NaN/Inf，发散时回退到动量
3. 使用 bfloat16 减少数值误差
4. 限制谱范数估计的范围

### 6.2 计算开销

**问题**: 谱范数估计和 NS 迭代增加计算

**对策**:
1. 谱范数估计只做 2 次幂迭代
2. NS 迭代固定 5 步，使用 bfloat16
3. 异步计算（如果可能）

**预期开销**: < 10% 的总训练时间

### 6.3 超参数调优

**问题**: 新引入的参数需要调优

**推荐值**:
- noise_scale: 0.001 - 0.01
- ns_steps: 5
- adaptive_ns: True
- hessian_approx: True

**调优策略**: 先用默认值，根据收敛情况微调 noise_scale。

---

## 七、总结

### 7.1 核心贡献

1. **自适应 NS 系数**: 根据谱范数动态调整，平衡收敛速度和稳定性
2. **混合更新策略**: 2D用正交化，1D/3D用Adam，各司其职
3. **Hessian近似**: 轻量级二阶信息，加速收敛
4. **噪声注入**: 帮助逃离局部最优
5. **内存优化**: 显存占用不超过 AdamW

### 7.2 预期效果

| 指标 | 预期改进 |
|-----|---------|
| 收敛速度 | 比 Muon 快 10-20% |
| 最终性能 | 持平或略优于 Muon |
| 稳定性 | 显著提高 |
| 泛化能力 | 更好（更宽的极小值）|
| 显存占用 | ≈ AdamW |

### 7.3 下一步工作

1. **实验验证**: 在标准 benchmark 上测试
2. **理论完善**: 证明收敛性，分析泛化界
3. **工程优化**: 实现高效的 CUDA kernel
4. **扩展应用**: 测试在 CV、NLP、RL 等任务上的效果

---

**文档版本**: v1.0  
**最后更新**: 2026-03-08  
**作者**: AI Agent + 苦力协作
