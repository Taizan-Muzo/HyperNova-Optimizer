"""
Configuration schemas for HyperNova using Pydantic.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class OptimizerConfig(BaseModel):
    """Core optimizer configuration supporting all model scales."""
    
    name: str = Field(default="adamw", description="Optimizer name: adamw, sgd, lion, etc.")
    lr: float = Field(default=1e-4, ge=0, description="Learning rate")
    betas: tuple = Field(default=(0.9, 0.999), description="Momentum coefficients")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    eps: float = Field(default=1e-8, gt=0, description="Numerical stability epsilon")
    maximize: bool = Field(default=False, description="Maximize objective")
    foreach: Optional[bool] = Field(default=None, description="Use foreach for performance")
    
    class Config:
        frozen = True


class LayerOptimizationStrategy(BaseModel):
    """
    Per-layer-type optimization strategy.
    
    Examples:
        - LayerNorm: weight_decay_scale=0.0 (no weight decay)
        - Embedding: lr_scale=10.0 (higher learning rate)
        - Attention: custom betas
    """
    
    class_name: str = Field(description="PyTorch module class name, e.g., 'LayerNorm', 'Linear'")
    lr_scale: float = Field(default=1.0, ge=0, description="Learning rate scale")
    weight_decay_scale: float = Field(default=1.0, ge=0, description="Weight decay scale")
    betas: Optional[tuple] = Field(default=None, description="Override betas for this layer type")
    
    class Config:
        protected_namespaces = ()


class TrainingEngineConfig(BaseModel):
    """
    Training engine global configuration.
    Supports all scales from tiny (MNIST) to massive (GPT-4).
    """
    
    optimizer_cfg: OptimizerConfig = Field(default_factory=OptimizerConfig)
    layer_strategy: List[LayerOptimizationStrategy] = Field(default_factory=list)
    
    # Training settings
    amp: bool = Field(default=True, description="Automatic Mixed Precision")
    grad_accum_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    max_grad_norm: Optional[float] = Field(default=None, ge=0, description="Max gradient norm for clipping")
    
    # Paths
    checkpoint_dir: str = Field(default="./checkpoints", description="Checkpoint directory")
    log_dir: str = Field(default="./logs", description="Log directory")
    
    # Device and distributed
    device: str = Field(default="auto", description="Device: auto, cuda, cpu")
    dist_backend: str = Field(default="nccl", description="Distributed backend")
    world_size: int = Field(default=1, ge=1, description="Number of processes for distributed training")
    
    # Memory optimization
    use_spectral_state: bool = Field(default=False, description="Use log-spectral state for memory saving")
    rank_ratio: float = Field(default=0.1, ge=0, le=1, description="Rank ratio for spectral compression")
    
    @validator('layer_strategy', pre=True, always=True)
    def set_default_strategy(cls, v):
        """Set default layer strategies if not provided."""
        if not v:
            return [
                LayerOptimizationStrategy(class_name="LayerNorm", weight_decay_scale=0.0),
                LayerOptimizationStrategy(class_name="BatchNorm", weight_decay_scale=0.0),
                LayerOptimizationStrategy(class_name="GroupNorm", weight_decay_scale=0.0),
            ]
        return v
    
    @validator('device', pre=True, always=True)
    def set_device(cls, v):
        """Auto-detect device if set to 'auto'."""
        if v == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v
