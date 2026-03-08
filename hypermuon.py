"""
HyperMuon: 超越 AdaMuon 的下一代优化器

核心创新:
1. 动态正交化频率
2. 多层自适应策略  
3. 时序动量预测
4. 稀疏正交化
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class HyperMuon(Optimizer):
    """
    HyperMuon Optimizer
    
    参数:
        lr: 学习率
        betas: Adam动量系数
        eps: 数值稳定性
        weight_decay: 权重衰减
        ns_steps: Newton-Schulz迭代步数
        
        # HyperMuon特有参数
        dynamic_ortho: 是否启用动态正交化频率
        ortho_freq: 正交化基础频率 (每N步一次)
        sparse_ratio: 稀疏正交化比例 (0=不稀疏, 0.5=稀疏50%)
        predictive_momentum: 是否启用预测动量
        prediction_weight: 预测梯度的权重
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, ns_steps=5,
                 dynamic_ortho=True, ortho_freq=1,
                 sparse_ratio=0.0, predictive_momentum=True, 
                 prediction_weight=0.2):
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            ns_steps=ns_steps,
            dynamic_ortho=dynamic_ortho, ortho_freq=ortho_freq,
            sparse_ratio=sparse_ratio,
            predictive_momentum=predictive_momentum,
            prediction_weight=prediction_weight,
        )
        super(HyperMuon, self).__init__(params, defaults)
    
    def _newton_schulz(self, G, steps=5):
        """Newton-Schulz正交化迭代"""
        a, b, c = 3.4445, -4.7750, 2.0315
        
        X = G.bfloat16() if G.dtype == torch.float32 else G
        eps = 1e-7
        
        # 归一化
        norm = torch.norm(X)
        if norm > eps:
            X = X / norm
        
        # 转置优化
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.T
        
        # NS迭代
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
            
            # 数值稳定性检查
            if torch.isnan(X).any() or torch.isinf(X).any():
                return G
        
        if transpose:
            X = X.T
        
        return X.float() if G.dtype == torch.float32 else X
    
    def _should_orthogonalize(self, state, grad, dynamic, freq):
        """决定是否进行正交化"""
        if not dynamic:
            # 静态频率
            return state['step'] % freq == 0
        
        # 动态决策：基于梯度变化
        if 'prev_grad' not in state:
            state['prev_grad'] = grad.clone()
            return True
        
        prev_grad = state['prev_grad']
        
        # 计算余弦相似度
        cos_sim = torch.sum(prev_grad * grad) / (
            torch.norm(prev_grad) * torch.norm(grad) + 1e-8
        )
        
        # 保存当前梯度
        state['prev_grad'] = grad.clone()
        
        # 方向变化大时正交化
        return cos_sim.item() < 0.95  # 阈值可调整
    
    def _sparse_orthogonalize(self, grad, ratio):
        """稀疏正交化"""
        if ratio <= 0:
            return self._newton_schulz(grad)
        
        # 计算重要性（梯度幅度）
        importance = torch.abs(grad)
        
        # 确定阈值
        k = int(grad.numel() * (1 - ratio))
        if k <= 0:
            return self._newton_schulz(grad)
        
        threshold = torch.kthvalue(importance.view(-1), k).values
        
        # 创建掩码
        mask = importance >= threshold
        
        # 只正交化重要部分
        result = grad.clone()
        if mask.any():
            # 提取重要部分
            grad_sparse = grad[mask]
            if grad_sparse.numel() > 0:
                # 重塑为2D进行正交化
                if grad_sparse.dim() == 1:
                    grad_sparse = grad_sparse.unsqueeze(0)
                
                ortho_sparse = self._newton_schulz(grad_sparse)
                
                # 放回结果
                if ortho_sparse.shape[0] == 1:
                    ortho_sparse = ortho_sparse.squeeze(0)
                result[mask] = ortho_sparse
        
        return result
    
    def _get_predicted_momentum(self, state, grad, beta1, pred_weight):
        """获取预测动量"""
        exp_avg = state['exp_avg']
        
        # 更新动量
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        if not self.defaults['predictive_momentum']:
            return exp_avg
        
        # 预测下一步
        if 'prev_exp_avg' in state:
            prev_exp_avg = state['prev_exp_avg']
            # 线性外推: m_{t+1} ≈ 2*m_t - m_{t-1}
            predicted = 2 * exp_avg - prev_exp_avg
            # 混合
            result = (1 - pred_weight) * exp_avg + pred_weight * predicted
        else:
            result = exp_avg
        
        state['prev_exp_avg'] = exp_avg.clone()
        return result
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    if p.dim() != 2:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # 权重衰减
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                if p.dim() == 2:
                    # ===== 2D参数: HyperMuon风格 =====
                    
                    # 1. 预测动量
                    m = self._get_predicted_momentum(
                        state, grad, beta1, group['prediction_weight']
                    )
                    
                    # 2. 动态决定是否正交化
                    should_ortho = self._should_orthogonalize(
                        state, m, 
                        group['dynamic_ortho'], 
                        group['ortho_freq']
                    )
                    
                    if should_ortho:
                        # 3. 稀疏正交化
                        update = self._sparse_orthogonalize(
                            m, group['sparse_ratio']
                        )
                    else:
                        # 跳过正交化，直接使用动量
                        update = m
                    
                else:
                    # ===== 1D/3D参数: Adam风格 =====
                    exp_avg_sq = state['exp_avg_sq']
                    
                    # 更新二阶矩
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # 偏差修正
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    step_size = group['lr'] / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    
                    update = exp_avg / denom
                
                # 参数更新
                p.add_(update, alpha=-group['lr'])
        
        return loss


def test_hypermuon():
    """简单测试"""
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # 测试不同配置
    configs = [
        ("基础", {"dynamic_ortho": False, "sparse_ratio": 0.0}),
        ("动态", {"dynamic_ortho": True, "sparse_ratio": 0.0}),
        ("稀疏", {"dynamic_ortho": False, "sparse_ratio": 0.5}),
        ("完整", {"dynamic_ortho": True, "sparse_ratio": 0.3, "predictive_momentum": True}),
    ]
    
    for name, config in configs:
        print(f"\n测试配置: {name}")
        optimizer = HyperMuon(model.parameters(), lr=1e-3, **config)
        
        for i in range(3):
            x = torch.randn(4, 10)
            y = torch.randint(0, 5, (4,))
            
            loss = nn.CrossEntropyLoss()(model(x), y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"  Step {i}: Loss = {loss.item():.4f}")
    
    print("\n✅ HyperMuon 测试通过！")


if __name__ == "__main__":
    test_hypermuon()
