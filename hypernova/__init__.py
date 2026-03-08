"""
HyperNova Optimizer Library

A production-grade optimizer library supporting all architectures (CNN/Transformer/Linear/LN)
from tiny to massive models.

Author: AI Agent + Collaborators
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "HyperNova Team"

from .optimizer.base import OptimizerBase
from .optimizer.registry import OptimizerRegistry
from .engine.trainer import ProductionTrainer
from .config.schema import (
    OptimizerConfig,
    LayerOptimizationStrategy,
    TrainingEngineConfig,
)

__all__ = [
    "OptimizerBase",
    "OptimizerRegistry",
    "ProductionTrainer",
    "OptimizerConfig",
    "LayerOptimizationStrategy",
    "TrainingEngineConfig",
]
