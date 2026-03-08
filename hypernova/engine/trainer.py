"""
Production training engine with support for:
- Mixed precision (AMP)
- Gradient accumulation
- Distributed training (DDP/FSDP ready)
- Gradient clipping
- Checkpointing
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Callable, Dict, Any
import os
from pathlib import Path


class ProductionTrainer:
    """
    Production-grade training engine.
    
    Supports all model scales from tiny (MNIST) to massive (GPT-4).
    """
    
    def __init__(self, model: nn.Module, config: Any):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            config: TrainingEngineConfig
        """
        self.model = model
        self.config = config
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Import here to avoid circular dependency
        from .optimizer.group_manager import LayerParamGroupManager
        from .optimizer.registry import OptimizerRegistry
        
        # Build parameter groups
        if config.layer_strategy:
            manager = LayerParamGroupManager(
                model, 
                config.layer_strategy,
                base_lr=config.optimizer_cfg.lr,
                base_weight_decay=config.optimizer_cfg.weight_decay,
                base_betas=config.optimizer_cfg.betas,
            )
            param_groups = manager.get_param_groups()
        else:
            param_groups = [{'params': model.parameters()}]
        
        # Create optimizer
        self.optimizer = OptimizerRegistry.create_optimizer(
            config.optimizer_cfg.name,
            param_groups,
            **config.optimizer_cfg.dict(exclude={'name'})
        )
        
        # Mixed precision
        self.amp_enabled = config.amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.amp_enabled else None
        
        # State tracking
        self.global_step = 0
        self.epoch = 0
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _forward_backward(
        self, 
        batch: Dict[str, torch.Tensor], 
        loss_fn: Callable,
        is_training: bool = True
    ) -> float:
        """
        Perform forward and backward pass.
        
        Args:
            batch: Input batch (dict with 'input' and 'target')
            loss_fn: Loss function
            is_training: Whether in training mode
        
        Returns:
            Loss value
        """
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        # Move batch to device
        input_data = batch['input'].to(self.device)
        target = batch['target'].to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.amp_enabled):
            output = self.model(input_data)
            loss = loss_fn(output, target)
        
        if is_training:
            # Scale loss for gradient accumulation
            loss = loss / self.config.grad_accum_steps
            
            # Backward pass
            if self.amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        return loss.item()
    
    def _update(self):
        """Perform optimizer step with gradient clipping."""
        # Gradient clipping
        if self.config.max_grad_norm:
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
        
        # Optimizer step
        if self.amp_enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.global_step += 1
    
    def train_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        loss_fn: Callable,
        batch_idx: int
    ) -> float:
        """
        Perform one training step.
        
        Args:
            batch: Input batch
            loss_fn: Loss function
            batch_idx: Current batch index (for gradient accumulation)
        
        Returns:
            Loss value
        """
        # Forward and backward
        loss = self._forward_backward(batch, loss_fn, is_training=True)
        
        # Check if we should update (gradient accumulation)
        should_update = (batch_idx + 1) % self.config.grad_accum_steps == 0
        
        if should_update:
            self._update()
        
        return loss
    
    def validate(self, dataloader, loss_fn: Callable) -> float:
        """
        Run validation.
        
        Args:
            dataloader: Validation dataloader
            loss_fn: Loss function
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                loss = self._forward_backward(batch, loss_fn, is_training=False)
                total_loss += loss
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, filename: Optional[str] = None, **metadata):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename (default: checkpoint_step_{global_step}.pt)
            **metadata: Additional metadata to save
        """
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config.dict(),
            'metadata': metadata
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train_epoch(self, dataloader, loss_fn: Callable, log_interval: int = 10):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            loss_fn: Loss function
            log_interval: Logging interval (batches)
        """
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_step(batch, loss_fn, batch_idx)
            epoch_loss += loss
            
            if batch_idx % log_interval == 0:
                print(f"Epoch {self.epoch} | Batch {batch_idx} | Step {self.global_step} | Loss: {loss:.4f}")
        
        self.epoch += 1
        
        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        return avg_loss
