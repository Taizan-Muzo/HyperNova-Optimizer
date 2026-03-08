"""
AdaMuon: Adaptive Muon Optimizer
改进版优化器，结合 Muon 的正交化思想和 Adam 的自适应性

改进点：
1. 自适应 Newton-Schulz 系数
2. 混合策略：2D参数用正交化，1D/3D用AdamW
3. 二阶信息近似
4. 噪声注入
5. 内存优化
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class AdaMuon(Optimizer):
    """
    AdaMuon Optimizer
    
    参数:
        lr: 学习率 (默认: 1e-3)
        betas: Adam的动量系数 (默认: (0.9, 0.999))
        eps: 数值稳定性常数 (默认: 1e-8)
        weight_decay: 权重衰减 (默认: 0.01)
        ns_steps: Newton-Schulz迭代步数 (默认: 5)
        adaptive_ns: 是否使用自适应NS系数 (默认: True)
        noise_scale: 噪声强度 (默认: 0.0，建议0.001-0.01)
        hessian_approx: 是否使用Hessian近似 (默认: True)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, ns_steps=5, adaptive_ns=True,
                 noise_scale=0.0, hessian_approx=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       ns_steps=ns_steps, adaptive_ns=adaptive_ns,
                       noise_scale=noise_scale, hessian_approx=hessian_approx)
        super(AdaMuon, self).__init__(params, defaults)
    
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
                    # 动量
                    state['exp_avg'] = torch.zeros_like(p)
                    # 二阶矩 (用于1D/3D参数)
                    if p.dim() != 2:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    # Hessian近似 (仅用于2D参数)
                    if p.dim() == 2 and group['hessian_approx']:
                        state['hessian_diag'] = torch.zeros(p.shape[0], device=p.device)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # 权重衰减 (AdamW风格)
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # 更新动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                if p.dim() == 2:
                    # ========== 2D参数：使用Muon风格的正交化更新 ==========
                    self._update_2d(p, exp_avg, state, group)
                else:
                    # ========== 1D/3D参数：使用Adam风格更新 ==========
                    self._update_adam_style(p, exp_avg, state, grad, group)
        
        return loss
    
    def _update_2d(self, p, exp_avg, state, group):
        """2D参数的Muon风格更新"""
        G = exp_avg
        
        # 计算谱范数用于自适应NS系数
        if group['adaptive_ns']:
            with torch.no_grad():
                # 使用幂迭代估计谱范数
                u = torch.randn(G.shape[0], 1, device=G.device, dtype=G.dtype)
                for _ in range(2):  # 2次迭代足够估计
                    u = G @ (G.T @ u)
                    u = u / (u.norm() + group['eps'])
                spectral_norm = (u.T @ G @ (G.T @ u)).sqrt().item()
                
                # 根据谱范数调整NS系数
                # 谱范数大时增加收敛速度，小时增加稳定性
                scale_factor = min(max(spectral_norm, 0.5), 2.0)
                a = 3.4445 * scale_factor
                b = -4.7750 * scale_factor
                c = 2.0315 * scale_factor
        else:
            a, b, c = 3.4445, -4.7750, 2.0315
        
        # Newton-Schulz迭代
        X = G.bfloat16() if G.dtype == torch.float32 else G
        
        # 归一化
        norm = X.norm()
        if norm > group['eps']:
            X = X / norm
        
        # 转置优化：如果行数大于列数，先转置
        transpose = False
        if X.shape[0] > X.shape[1]:
            X = X.T
            transpose = True
        
        # NS迭代
        for _ in range(group['ns_steps']):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
            
            # 数值稳定性检查
            if torch.isnan(X).any() or torch.isinf(X).any():
                # 如果发散，回退到简单动量更新
                X = G
                break
        
        if transpose:
            X = X.T
        
        # 转回原始精度
        if G.dtype == torch.float32 and X.dtype == torch.bfloat16:
            X = X.float()
        
        # Hessian近似：自适应学习率缩放
        if group['hessian_approx']:
            # 计算梯度外积的对角线近似
            hessian_diag = state.get('hessian_diag')
            if hessian_diag is not None:
                # H_ii ≈ E[g_i^2]
                grad_sq = (G ** 2).mean(dim=1)
                hessian_diag.mul_(0.9).add_(grad_sq, alpha=0.1)
                
                # 使用Hessian对角线调整学习率
                # 相当于对每个输出维度做AdaGrad
                adaptive_lr = 1.0 / (hessian_diag.sqrt() + group['eps'])
                adaptive_lr = adaptive_lr.unsqueeze(1)
                
                # 应用自适应缩放
                X = X * adaptive_lr.clamp_max(10.0)  # 防止过大
        
        # 噪声注入
        if group['noise_scale'] > 0:
            noise = torch.randn_like(X) * group['noise_scale'] * X.std()
            X = X + noise
        
        # 更新参数
        p.add_(X, alpha=-group['lr'])
    
    def _update_adam_style(self, p, exp_avg, state, grad, group):
        """1D/3D参数的Adam风格更新"""
        exp_avg_sq = state['exp_avg_sq']
        beta1, beta2 = group['betas']
        
        # 更新二阶矩
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # 偏差修正
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        step_size = group['lr'] / bias_correction1
        
        # Adam更新
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        p.addcdiv_(exp_avg, denom, value=-step_size)
    
    def _newton_schulz5_adaptive(self, G, steps, eps, spectral_norm):
        """
        自适应Newton-Schulz迭代
        根据谱范数动态调整系数
        """
        # 根据谱范数调整系数
        # 谱范数大时：增加a加速收敛
        # 谱范数小时：减小a提高稳定性
        scale = min(max(spectral_norm, 0.5), 2.0)
        a = 3.4445 * scale
        b = -4.7750 * scale  
        c = 2.0315 * scale
        
        X = G.bfloat16()
        X /= (X.norm() + eps)
        
        if G.size(0) > G.size(1):
            X = X.T
        
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if G.size(0) > G.size(1):
            X = X.T
        
        return X


def test_adamuon():
    """简单测试"""
    import torch.nn as nn
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    
    # 使用AdaMuon
    optimizer = AdaMuon(model.parameters(), lr=1e-3, noise_scale=0.001)
    
    # 模拟训练
    for step in range(10):
        x = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))
        
        loss = nn.CrossEntropyLoss()(model(x), y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: Loss = {loss.item():.4f}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_adamuon()
