#!/usr/bin/env python3
"""
LLaMA 预训练对比实验
对比 AdamW 和 HyperNova 在语言模型训练上的表现
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.experiment_framework import ExperimentConfig
import json
import time
from typing import Dict, List


class DummyTextDataset(Dataset):
    """模拟文本数据集用于测试"""
    
    def __init__(self, num_samples=10000, seq_len=512, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机token序列
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }


def get_llama_model(model_size='7b'):
    """
    创建 LLaMA 模型
    由于实际 LLaMA 模型很大，这里使用简化的 Transformer 作为代理
    """
    if model_size == '7b':
        # LLaMA-7B 配置
        config = {
            'vocab_size': 32000,
            'hidden_size': 4096,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'intermediate_size': 11008,
            'max_position_embeddings': 2048,
        }
    elif model_size == '13b':
        config = {
            'vocab_size': 32000,
            'hidden_size': 5120,
            'num_hidden_layers': 40,
            'num_attention_heads': 40,
            'intermediate_size': 13824,
            'max_position_embeddings': 2048,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # 使用简化的 Transformer 作为代理
    from transformers import LlamaConfig, LlamaForCausalLM
    
    llama_config = LlamaConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        max_position_embeddings=config['max_position_embeddings'],
    )
    
    model = LlamaForCausalLM(llama_config)
    return model


def create_dummy_llama_model(hidden_size=512, num_layers=4, num_heads=8):
    """创建小型 LLaMA 模型用于快速测试"""
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
        
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
        )
        
        model = LlamaForCausalLM(config)
        return model
    except ImportError:
        print("transformers not installed, using simple transformer")
        # 回退到简单模型
        return SimpleTransformer(
            vocab_size=1000,
            d_model=hidden_size,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=hidden_size * 4,
        )


class SimpleTransformer(nn.Module):
    """简化版 Transformer 用于测试"""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids) + self.pos_encoding[:, :input_ids.size(1), :]
        x = self.transformer(x)
        logits = self.output(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {'logits': logits, 'loss': loss}


class LLaMAExperiment:
    """LLaMA 训练实验"""
    
    def __init__(self, config: ExperimentConfig, model_size='tiny'):
        self.config = config
        self.model_size = model_size
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    def get_optimizer(self, model):
        """创建优化器"""
        name = self.config.optimizer_name.lower()
        
        if name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        
        elif name == "hypernova":
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
    
    def train_step(self, model, batch, optimizer):
        """单步训练"""
        model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        return loss.item()
    
    def run_experiment(self, num_steps=1000, log_interval=100):
        """运行实验"""
        print(f"\n{'='*80}")
        print(f"LLaMA Training Experiment: {self.config.optimizer_name}")
        print(f"Model size: {self.model_size}")
        print(f"{'='*80}\n")
        
        # 创建模型
        if self.model_size == 'tiny':
            model = create_dummy_llama_model(hidden_size=256, num_layers=4, num_heads=8)
        elif self.model_size == 'small':
            model = create_dummy_llama_model(hidden_size=512, num_layers=6, num_heads=8)
        elif self.model_size == '7b':
            model = get_llama_model('7b')
        else:
            raise ValueError(f"Unknown model size: {self.model_size}")
        
        model = model.to(self.device)
        optimizer = self.get_optimizer(model)
        
        # 创建数据集
        dataset = DummyTextDataset(
            num_samples=num_steps * self.config.batch_size,
            seq_len=512,
            vocab_size=32000 if self.model_size in ['7b', '13b'] else 1000
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 记录指标
        losses = []
        tokens_per_sec = []
        
        # 记录内存
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        data_iter = iter(dataloader)
        
        for step in range(num_steps):
            step_start = time.time()
            
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            loss = self.train_step(model, batch, optimizer)
            losses.append(loss)
            
            step_time = time.time() - step_start
            tokens_sec = (self.config.batch_size * 512) / step_time
            tokens_per_sec.append(tokens_sec)
            
            if step % log_interval == 0:
                avg_loss = sum(losses[-log_interval:]) / min(log_interval, len(losses))
                avg_tokens = sum(tokens_per_sec[-log_interval:]) / min(log_interval, len(tokens_per_sec))
                print(f"Step {step:5d}/{num_steps} | Loss: {avg_loss:.4f} | Tokens/s: {avg_tokens:.0f}")
        
        total_time = time.time() - start_time
        
        # 获取峰值内存
        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024  # GB
        
        # 计算指标
        avg_loss = sum(losses) / len(losses)
        avg_tokens_per_sec = sum(tokens_per_sec) / len(tokens_per_sec)
        
        result = {
            'optimizer': self.config.optimizer_name,
            'model_size': self.model_size,
            'num_steps': num_steps,
            'avg_loss': avg_loss,
            'final_loss': losses[-1],
            'avg_tokens_per_sec': avg_tokens_per_sec,
            'total_time': total_time,
            'peak_memory_gb': peak_memory,
        }
        
        print(f"\n{'='*80}")
        print(f"Experiment completed!")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Final Loss: {losses[-1]:.4f}")
        print(f"  Avg Tokens/s: {avg_tokens_per_sec:.0f}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Peak Memory: {peak_memory:.2f} GB")
        print(f"{'='*80}\n")
        
        return result


def run_llama_experiments():
    """运行 LLaMA 对比实验"""
    print("="*80)
    print("LLaMA 训练对比实验")
    print("="*80)
    
    # 实验配置
    experiments = [
        {
            'name': 'adamw',
            'config': ExperimentConfig(
                optimizer_name='adamw',
                lr=1e-4,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                batch_size=4,
                device='cuda',
            )
        },
        {
            'name': 'hypernova',
            'config': ExperimentConfig(
                optimizer_name='hypernova',
                lr=1e-4,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                rank_ratio=0.1,
                use_gnn=True,
                batch_size=4,
                device='cuda',
            )
        },
    ]
    
    results = []
    
    for exp in experiments:
        runner = LLaMAExperiment(exp['config'], model_size='tiny')
        result = runner.run_experiment(num_steps=500, log_interval=50)
        result['name'] = exp['name']
        results.append(result)
    
    # 保存结果
    os.makedirs('./experiments/llama/results', exist_ok=True)
    with open('./experiments/llama/results/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印对比
    print("\n" + "="*80)
    print("LLaMA 实验结果对比")
    print("="*80)
    print(f"{'Optimizer':<15} {'Avg Loss':>10} {'Tokens/s':>12} {'Memory(GB)':>12}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<15} {r['avg_loss']:>10.4f} {r['avg_tokens_per_sec']:>12.0f} {r['peak_memory_gb']:>12.2f}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    run_llama_experiments()
