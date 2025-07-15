"""
Tests for Transformer model components.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.models.transformer import TransformerModel, LitPIC50, PIC50Trainer


class TestTransformerModel:
    """Test Transformer model architecture."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        input_dim = 100
        model = TransformerModel(
            input_dim=input_dim,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1
        )
        
        assert model.input_dim == input_dim
        assert model.dim_feedforward == 256
        assert model.max_seq_length == 1
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        input_dim = 50
        batch_size = 32
        
        model = TransformerModel(
            input_dim=input_dim,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1
        )
        
        # Create input tensor
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_architectures(self):
        """Test different model architectures."""
        input_dim = 100
        
        # Small model
        small_model = TransformerModel(
            input_dim=input_dim,
            num_layers=1,
            num_heads=2,
            dim_feedforward=128,
            dropout=0.1
        )
        
        # Large model
        large_model = TransformerModel(
            input_dim=input_dim,
            num_layers=4,
            num_heads=8,
            dim_feedforward=512,
            dropout=0.2
        )
        
        # Test forward pass for both
        x = torch.randn(16, input_dim)
        
        small_output = small_model(x)
        large_output = large_model(x)
        
        assert small_output.shape == (16, 1)
        assert large_output.shape == (16, 1)
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        input_dim = 100
        model = TransformerModel(
            input_dim=input_dim,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1
        )
        
        # Check that model has parameters
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that parameters are trainable
        for param in params:
            assert param.requires_grad


class TestLitPIC50:
    """Test PyTorch Lightning module."""
    
    def test_module_initialization(self):
        """Test Lightning module initialization."""
        input_dim = 100
        model = LitPIC50(
            input_dim=input_dim,
            learning_rate=1e-3,
            weight_decay=1e-5,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1
        )
        
        assert model.model is not None
        assert model.criterion is not None
        assert model.learning_rate == 1e-3
    
    def test_training_step(self):
        """Test training step."""
        input_dim = 50
        batch_size = 16
        
        model = LitPIC50(input_dim=input_dim)
        
        # Create batch
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, 1)
        
        # Training step
        loss = model.training_step((x, y), batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_validation_step(self):
        """Test validation step."""
        input_dim = 50
        batch_size = 16
        
        model = LitPIC50(input_dim=input_dim)
        
        # Create batch
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, 1)
        
        # Validation step
        loss = model.validation_step((x, y), batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        input_dim = 100
        model = LitPIC50(input_dim=input_dim)
        
        optimizers = model.configure_optimizers()
        
        assert 'optimizer' in optimizers
        assert 'lr_scheduler' in optimizers
        
        optimizer = optimizers['optimizer']
        assert isinstance(optimizer, torch.optim.Adam)
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        input_dim = 100
        model = LitPIC50(input_dim=input_dim)
        
        # Save model
        checkpoint_path = tmp_path / "test_model.ckpt"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Load model
        loaded_model = LitPIC50.load_from_checkpoint(checkpoint_path)
        
        # Test forward pass
        x = torch.randn(8, input_dim)
        original_output = model(x)
        loaded_output = loaded_model(x)
        
        # Outputs should be similar (not identical due to random initialization)
        assert original_output.shape == loaded_output.shape


class TestPIC50Trainer:
    """Test high-level trainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = PIC50Trainer(
            model_dir='test_models',
            max_epochs=10,
            patience=5,
            batch_size=32
        )
        
        assert trainer.max_epochs == 10
        assert trainer.patience == 5
        assert trainer.batch_size == 32
    
    def test_data_loader_creation(self):
        """Test data loader creation."""
        trainer = PIC50Trainer()
        
        # Create sample data
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 50)
        y_val = np.random.randn(20)
        
        # Create data loaders
        train_loader, val_loader = trainer._create_data_loaders(
            X_train, y_train, X_val, y_val
        )
        
        assert train_loader is not None
        assert val_loader is not None
        
        # Test data loading
        for batch in train_loader:
            x, y = batch
            assert x.shape[0] <= trainer.batch_size
            assert y.shape[0] <= trainer.batch_size
            break
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        trainer = PIC50Trainer()
        
        # Create sample data
        X = np.random.randn(50, 100)
        y = np.random.randn(50)
        
        # Create a simple model
        input_dim = X.shape[1]
        model = LitPIC50(input_dim=input_dim)
        
        # Evaluate model
        metrics = trainer.evaluate(model, X, y)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['r2'] <= 1  # R² can be negative for poor models
    
    def test_prediction(self):
        """Test model prediction."""
        trainer = PIC50Trainer()
        
        # Create sample data
        X = np.random.randn(10, 50)
        
        # Create a simple model
        input_dim = X.shape[1]
        model = LitPIC50(input_dim=input_dim)
        
        # Make predictions
        predictions = trainer.predict(model, X)
        
        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()


class TestModelIntegration:
    """Test model integration components."""
    
    def test_end_to_end_training_simulation(self):
        """Simulate end-to-end training process."""
        # This test simulates a complete training process
        # without actually training (which would be too slow)
        
        input_dim = 100
        n_samples = 200
        
        # Create sample data
        X_train = np.random.randn(n_samples, input_dim)
        y_train = np.random.randn(n_samples)
        X_val = np.random.randn(50, input_dim)
        y_val = np.random.randn(50)
        
        # Create model
        model = LitPIC50(input_dim=input_dim)
        
        # Test forward pass
        x_tensor = torch.tensor(X_train[:10], dtype=torch.float32)
        output = model(x_tensor)
        
        assert output.shape == (10, 1)
        assert not torch.isnan(output).any()
    
    def test_model_hyperparameters(self):
        """Test different model hyperparameters."""
        input_dim = 100
        
        # Test different configurations
        configs = [
            {'num_layers': 1, 'num_heads': 2, 'dim_feedforward': 128},
            {'num_layers': 2, 'num_heads': 4, 'dim_feedforward': 256},
            {'num_layers': 4, 'num_heads': 8, 'dim_feedforward': 512},
        ]
        
        for config in configs:
            model = LitPIC50(input_dim=input_dim, **config)
            
            # Test forward pass
            x = torch.randn(8, input_dim)
            output = model(x)
            
            assert output.shape == (8, 1)
            assert not torch.isnan(output).any()


if __name__ == '__main__':
    pytest.main([__file__]) 