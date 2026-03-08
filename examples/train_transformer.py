"""
Example: Training a Transformer model with HyperNova

This example demonstrates:
1. Layer-wise optimization strategies
2. Mixed precision training
3. Gradient accumulation
4. Checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hypernova import (
    ProductionTrainer,
    TrainingEngineConfig,
    OptimizerConfig,
    LayerOptimizationStrategy,
)


def create_dummy_data(num_samples=1000, seq_len=32, vocab_size=1000):
    """Create dummy dataset for demonstration."""
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def main():
    # Create a simple Transformer model
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        ),
        num_layers=4,
    )
    
    # Add embedding and output layers
    class TransformerClassifier(nn.Module):
        def __init__(self, encoder, vocab_size=1000, num_classes=10):
            super().__init__()
            self.encoder = encoder
            self.embedding = nn.Embedding(vocab_size, 128)
            self.output = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            # Use mean pooling
            x = x.mean(dim=1)
            return self.output(x)
    
    model = TransformerClassifier(model)
    
    # Configure training with layer-wise strategies
    config = TrainingEngineConfig(
        optimizer_cfg=OptimizerConfig(
            name="adamw",
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        layer_strategy=[
            # No weight decay for normalization layers
            LayerOptimizationStrategy(class_name="LayerNorm", weight_decay_scale=0.0),
            
            # Higher learning rate for embeddings
            LayerOptimizationStrategy(class_name="Embedding", lr_scale=5.0),
            
            # Lower learning rate for encoder layers
            LayerOptimizationStrategy(class_name="TransformerEncoderLayer", lr_scale=0.5),
        ],
        
        # Training settings
        amp=True,  # Enable mixed precision
        grad_accum_steps=2,  # Gradient accumulation
        max_grad_norm=1.0,  # Gradient clipping
        checkpoint_dir="./checkpoints/transformer_example",
    )
    
    # Create trainer
    trainer = ProductionTrainer(model, config)
    
    # Print parameter group info
    from hypernova.optimizer.group_manager import LayerParamGroupManager
    manager = LayerParamGroupManager(
        model,
        config.layer_strategy,
        base_lr=config.optimizer_cfg.lr,
        base_weight_decay=config.optimizer_cfg.weight_decay,
    )
    manager.print_group_info()
    
    # Create data
    train_loader = create_dummy_data(num_samples=500)
    val_loader = create_dummy_data(num_samples=100)
    
    # Training loop
    num_epochs = 5
    loss_fn = nn.CrossEntropyLoss()
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, loss_fn, log_interval=5)
        
        # Validate
        val_loss = trainer.validate(val_loader, loss_fn)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            trainer.save_checkpoint(epoch=epoch, val_loss=val_loss)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Final checkpoint
    trainer.save_checkpoint(final=True, train_loss=train_loss, val_loss=val_loss)


if __name__ == "__main__":
    main()
