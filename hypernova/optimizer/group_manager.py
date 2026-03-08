"""
Layer-wise parameter grouping for architecture-aware optimization.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Iterator, Tuple
import re


class LayerParamGroupManager:
    """
    Automatically group parameters by layer type for different optimization strategies.
    
    Supports all layer types:
    - Linear/Dense (fully connected)
    - Conv1D/2D/3D (convolutional)
    - LayerNorm/BatchNorm/GroupNorm (normalization)
    - Embedding (embedding layers)
    - Attention (MultiheadAttention, custom attention)
    - Any custom layers
    
    Usage:
        strategies = [
            LayerOptimizationStrategy(class_name="LayerNorm", weight_decay_scale=0.0),
            LayerOptimizationStrategy(class_name="Embedding", lr_scale=10.0),
        ]
        manager = LayerParamGroupManager(model, strategies)
        param_groups = manager.get_param_groups()
        optimizer = AdamW(param_groups, lr=1e-3)
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        strategies: List[Any],
        base_lr: float = 1e-4,
        base_weight_decay: float = 0.01,
        base_betas: Tuple[float, float] = (0.9, 0.999),
    ):
        """
        Initialize the parameter group manager.
        
        Args:
            model: PyTorch model
            strategies: List of layer optimization strategies
            base_lr: Base learning rate
            base_weight_decay: Base weight decay
            base_betas: Base betas for Adam-style optimizers
        """
        self.model = model
        self.strategies = {s.class_name: s for s in strategies}
        self.base_lr = base_lr
        self.base_weight_decay = base_weight_decay
        self.base_betas = base_betas
        self._param_groups: List[Dict[str, Any]] = []
        self._build_groups()
    
    def _match_strategy(self, module: nn.Module) -> Tuple[bool, Any]:
        """
        Check if a module matches any strategy.
        
        Matching rules (in order of priority):
        1. Exact class name match
        2. Partial class name match (case-insensitive)
        3. Parent class match (e.g., '_NormBase' matches all norms)
        """
        module_name = type(module).__name__
        module_base = type(module).__base__.__name__ if type(module).__base__ else ""
        
        # 1. Exact match
        if module_name in self.strategies:
            return True, self.strategies[module_name]
        
        # 2. Partial match (case-insensitive)
        for name, strategy in self.strategies.items():
            if name.lower() in module_name.lower() or module_name.lower() in name.lower():
                return True, strategy
        
        # 3. Check for norm layers (common base)
        if 'Norm' in module_name and 'Norm' in str(self.strategies.keys()):
            for name, strategy in self.strategies.items():
                if 'Norm' in name:
                    return True, strategy
        
        return False, None
    
    def _build_groups(self):
        """Build parameter groups based on layer types."""
        if not self.strategies:
            # No custom strategies, use default single group
            self._param_groups = [{
                'params': [p for p in self.model.parameters() if p.requires_grad],
                'lr': self.base_lr,
                'weight_decay': self.base_weight_decay,
                'betas': self.base_betas,
                'layer_type': 'default'
            }]
            return
        
        # Track which parameters have been assigned
        assigned_params = set()
        
        # Group parameters by layer type
        type_to_params: Dict[str, List[torch.nn.Parameter]] = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                # Skip container modules (they don't have parameters directly)
                continue
            
            # Get parameters of this module
            module_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            if not module_params:
                continue
            
            # Check if already assigned (avoid duplicates)
            module_params = [p for p in module_params if id(p) not in assigned_params]
            if not module_params:
                continue
            
            # Match strategy
            matched, strategy = self._match_strategy(module)
            
            if matched:
                layer_type = type(module).__name__
                if layer_type not in type_to_params:
                    type_to_params[layer_type] = {
                        'params': [],
                        'strategy': strategy
                    }
                type_to_params[layer_type]['params'].extend(module_params)
                for p in module_params:
                    assigned_params.add(id(p))
        
        # Build parameter groups from type_to_params
        for layer_type, info in type_to_params.items():
            strategy = info['strategy']
            group = {
                'params': info['params'],
                'lr': self.base_lr * strategy.lr_scale,
                'weight_decay': self.base_weight_decay * strategy.weight_decay_scale,
                'betas': strategy.betas if strategy.betas else self.base_betas,
                'layer_type': layer_type,
            }
            self._param_groups.append(group)
        
        # Add remaining unassigned parameters to default group
        remaining_params = [
            p for p in self.model.parameters() 
            if p.requires_grad and id(p) not in assigned_params
        ]
        
        if remaining_params:
            self._param_groups.append({
                'params': remaining_params,
                'lr': self.base_lr,
                'weight_decay': self.base_weight_decay,
                'betas': self.base_betas,
                'layer_type': 'default'
            })
    
    def get_param_groups(self) -> List[Dict[str, Any]]:
        """Return the constructed parameter groups."""
        return self._param_groups
    
    def print_group_info(self):
        """Print information about parameter groups (for debugging)."""
        print("=" * 60)
        print("Parameter Group Information")
        print("=" * 60)
        
        for i, group in enumerate(self._param_groups):
            num_params = sum(p.numel() for p in group['params'])
            print(f"\nGroup {i}: {group['layer_type']}")
            print(f"  Parameters: {num_params:,}")
            print(f"  LR: {group['lr']:.2e}")
            print(f"  Weight Decay: {group['weight_decay']:.2e}")
            print(f"  Betas: {group['betas']}")
        
        total_params = sum(sum(p.numel() for p in g['params']) for g in self._param_groups)
        print(f"\nTotal Parameters: {total_params:,}")
        print("=" * 60)
