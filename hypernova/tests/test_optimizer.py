"""
Unit tests for HyperNova Optimizer
"""

import pytest
import torch
import torch.nn as nn
from hypernova import (
    ProductionTrainer,
    TrainingEngineConfig,
    OptimizerConfig,
    LayerOptimizationStrategy,
)


class TinyModel(nn.Module):
    """Simple test model with multiple layer types."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.linear1 = nn.Linear(32, 64)
        self.norm = nn.LayerNorm(64)
        self.linear2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.linear1(x))
        x = self.norm(x)
        x = self.linear2(x)
        return x


class CNNModel(nn.Module):
    """CNN model for testing Conv layers."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.norm(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def tiny_model():
    return TinyModel()


@pytest.fixture
def cnn_model():
    return CNNModel()


@pytest.fixture
def config():
    return TrainingEngineConfig(
        optimizer_cfg=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01),
        layer_strategy=[
            LayerOptimizationStrategy(class_name="LayerNorm", weight_decay_scale=0.0),
            LayerOptimizationStrategy(class_name="BatchNorm", weight_decay_scale=0.0),
            LayerOptimizationStrategy(class_name="Embedding", lr_scale=10.0),
        ],
        amp=False,
        grad_accum_steps=1,
    )


def test_param_grouping(tiny_model, config):
    """Test that parameter grouping works correctly."""
    from hypernova.optimizer.group_manager import LayerParamGroupManager
    
    manager = LayerParamGroupManager(
        tiny_model,
        config.layer_strategy,
        base_lr=config.optimizer_cfg.lr,
        base_weight_decay=config.optimizer_cfg.weight_decay,
    )
    
    groups = manager.get_param_groups()
    
    # Should have at least 3 groups: Embedding, Linear, Norm, and default
    assert len(groups) >= 3
    
    # Check that embedding has higher LR
    embedding_group = [g for g in groups if g['layer_type'] == 'Embedding'][0]
    assert embedding_group['lr'] == config.optimizer_cfg.lr * 10.0
    
    # Check that norm has no weight decay
    norm_groups = [g for g in groups if 'Norm' in g['layer_type']]
    for group in norm_groups:
        assert group['weight_decay'] == 0.0


def test_trainer_initialization(tiny_model, config):
    """Test trainer initialization."""
    trainer = ProductionTrainer(tiny_model, config)
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.global_step == 0
    assert trainer.epoch == 0


def test_training_step(tiny_model, config):
    """Test single training step."""
    trainer = ProductionTrainer(tiny_model, config)
    
    # Create dummy batch
    batch = {
        'input': torch.randint(0, 100, (4, 10)),  # batch_size=4, seq_len=10
        'target': torch.randint(0, 10, (4,)),
    }
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Run training step
    loss = trainer.train_step(batch, loss_fn, batch_idx=0)
    
    assert loss > 0
    assert isinstance(loss, float)


def test_validation(tiny_model, config):
    """Test validation."""
    trainer = ProductionTrainer(tiny_model, config)
    
    # Create dummy dataloader
    class DummyDataLoader:
        def __iter__(self):
            for _ in range(2):
                yield {
                    'input': torch.randint(0, 100, (4, 10)),
                    'target': torch.randint(0, 10, (4,)),
                }
        
        def __len__(self):
            return 2
    
    val_loader = DummyDataLoader()
    loss_fn = nn.CrossEntropyLoss()
    
    val_loss = trainer.validate(val_loader, loss_fn)
    
    assert val_loss >= 0
    assert isinstance(val_loss, float)


def test_checkpoint_save_load(tiny_model, config, tmp_path):
    """Test checkpoint saving and loading."""
    trainer = ProductionTrainer(tiny_model, config)
    
    # Do one step to change state
    batch = {
        'input': torch.randint(0, 100, (4, 10)),
        'target': torch.randint(0, 10, (4,)),
    }
    loss_fn = nn.CrossEntropyLoss()
    trainer.train_step(batch, loss_fn, 0)
    
    # Save checkpoint
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    trainer.save_checkpoint(filename="test_checkpoint.pt", test_metric=0.95)
    
    # Verify file exists
    assert checkpoint_path.exists()
    
    # Load checkpoint
    trainer2 = ProductionTrainer(tiny_model, config)
    trainer2.load_checkpoint(str(checkpoint_path))
    
    # Verify state restored
    assert trainer2.global_step == trainer.global_step
    assert trainer2.epoch == trainer.epoch


def test_cnn_model(cnn_model, config):
    """Test with CNN model."""
    trainer = ProductionTrainer(cnn_model, config)
    
    batch = {
        'input': torch.randn(4, 3, 32, 32),
        'target': torch.randint(0, 10, (4,)),
    }
    loss_fn = nn.CrossEntropyLoss()
    
    loss = trainer.train_step(batch, loss_fn, batch_idx=0)
    
    assert loss > 0


def test_gradient_accumulation(tiny_model, config):
    """Test gradient accumulation."""
    config.grad_accum_steps = 4
    trainer = ProductionTrainer(tiny_model, config)
    
    batch = {
        'input': torch.randint(0, 100, (4, 10)),
        'target': torch.randint(0, 10, (4,)),
    }
    loss_fn = nn.CrossEntropyLoss()
    
    # Run 4 steps (should accumulate gradients)
    losses = []
    for i in range(4):
        loss = trainer.train_step(batch, loss_fn, i)
        losses.append(loss)
    
    # Optimizer should have updated only once (at step 3)
    assert trainer.global_step == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
