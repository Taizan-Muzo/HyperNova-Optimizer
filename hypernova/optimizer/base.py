"""
Base optimizer classes and registry.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Type, Dict, TypeVar, Optional, Any, List
from abc import ABC, abstractmethod

T = TypeVar('T', bound='OptimizerBase')


class OptimizerBase(Optimizer):
    """
    Abstract base class for all HyperNova optimizers.
    
    Features:
    - Step counting and logging hooks
    - State checkpointing support
    - Learning rate scheduling compatibility
    """
    
    def __init__(self, params, **defaults):
        super().__init__(params, **defaults)
        self._step_count = 0
        self._loss_history = []
        self._lr_scheduler = None
    
    @property
    def step_count(self) -> int:
        """Return the number of steps taken."""
        return self._step_count
    
    def step(self, closure=None):
        """Perform a single optimization step with hooks."""
        self._step_count += 1
        
        # Pre-step hook (can be overridden)
        self._pre_step_hook()
        
        # Perform optimization step
        loss = super().step(closure)
        
        # Post-step hook (can be overridden)
        self._post_step_hook()
        
        return loss
    
    def _pre_step_hook(self):
        """Hook called before step. Override for custom behavior."""
        pass
    
    def _post_step_hook(self):
        """Hook called after step. Override for custom behavior."""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict with additional metadata."""
        state = super().state_dict()
        state['step_count'] = self._step_count
        state['loss_history'] = self._loss_history
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict with metadata."""
        self._step_count = state_dict.get('step_count', 0)
        self._loss_history = state_dict.get('loss_history', [])
        super().load_state_dict(state_dict)


class OptimizerRegistry:
    """
    Factory registry for optimizers.
    
    Usage:
        @OptimizerRegistry.register("adamw")
        class AdamW(OptimizerBase):
            ...
        
        optimizer = OptimizerRegistry.create_optimizer("adamw", params, lr=1e-3)
    """
    
    _registry: Dict[str, Type[OptimizerBase]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an optimizer class."""
        def decorator(klass: Type[T]) -> Type[T]:
            cls._registry[name.lower()] = klass
            return klass
        return decorator
    
    @classmethod
    def create_optimizer(
        cls, 
        name: str, 
        param_groups: List[Dict[str, Any]], 
        **kwargs
    ) -> OptimizerBase:
        """
        Create an optimizer instance by name.
        
        Args:
            name: Optimizer name (e.g., 'adamw', 'sgd')
            param_groups: Parameter groups from LayerParamGroupManager
            **kwargs: Optimizer-specific arguments
        
        Returns:
            Optimizer instance
        """
        name = name.lower()
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Unknown optimizer '{name}'. Available: {available}")
        
        klass = cls._registry[name]
        return klass(param_groups, **kwargs)
    
    @classmethod
    def list_optimizers(cls) -> List[str]:
        """List all registered optimizer names."""
        return list(cls._registry.keys())
    
    @classmethod
    def get_optimizer_class(cls, name: str) -> Optional[Type[OptimizerBase]]:
        """Get optimizer class by name."""
        return cls._registry.get(name.lower())
