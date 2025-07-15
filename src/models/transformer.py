"""
Transformer-based pIC50 prediction model using PyTorch Lightning.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from ..utils.config import config


class TransformerModel(nn.Module):
    """Transformer-based model for pIC50 prediction."""
    
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 1
    ):
        """Initialize the transformer model.
        
        Args:
            input_dim: Input feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length (1 for single molecules)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.dim_feedforward = dim_feedforward
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, dim_feedforward)
        
        # Positional encoding (simple for single molecules)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, dim_feedforward))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(dim_feedforward, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Project input to transformer dimension
        x = self.input_projection(x)  # (batch_size, dim_feedforward)
        
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1)  # (batch_size, 1, dim_feedforward)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 1, dim_feedforward)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch_size, dim_feedforward, 1)
        x = self.global_pool(x)  # (batch_size, dim_feedforward, 1)
        x = x.squeeze(-1)  # (batch_size, dim_feedforward)
        
        # Output projection
        x = self.dropout(x)
        x = self.output_projection(x)  # (batch_size, 1)
        
        return x


class LitPIC50(pl.LightningModule):
    """PyTorch Lightning module for pIC50 prediction."""
    
    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1
    ):
        """Initialize the Lightning module.
        
        Args:
            input_dim: Input feature dimension
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            scheduler_step_size: Learning rate scheduler step size
            scheduler_gamma: Learning rate scheduler gamma
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = TransformerModel(
            input_dim=input_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Learning rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.train_r2_scores = []
        self.val_r2_scores = []
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Tuple of (features, targets)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate R² score
        y_np = y.cpu().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()
        r2 = r2_score(y_np, y_hat_np)
        self.log('train_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (features, targets)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate R² score
        y_np = y.cpu().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()
        r2 = r2_score(y_np, y_hat_np)
        self.log('val_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step.
        
        Args:
            batch: Tuple of (features, targets)
            batch_idx: Batch index
            
        Returns:
            Dictionary with test metrics
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        y_np = y.cpu().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()
        r2 = r2_score(y_np, y_hat_np)
        rmse = np.sqrt(mean_squared_error(y_np, y_hat_np))
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_r2', r2)
        self.log('test_rmse', rmse)
        
        return {
            'test_loss': loss,
            'test_r2': torch.tensor(r2),
            'test_rmse': torch.tensor(rmse)
        }
    
    def configure_optimizers(self) -> Dict:
        """Configure optimizers and schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Prediction step.
        
        Args:
            batch: Input tensor
            batch_idx: Batch index
            
        Returns:
            Predictions tensor
        """
        return self(batch)


class PIC50Trainer:
    """High-level trainer for pIC50 prediction models."""
    
    def __init__(
        self,
        model_dir: str = 'models',
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        """Initialize the trainer.
        
        Args:
            model_dir: Directory to save models
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            batch_size: Batch size
            num_workers: Number of data loader workers
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.logger = logging.getLogger(__name__)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str,
        **model_kwargs
    ) -> LitPIC50:
        """Train a pIC50 prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name for the model
            **model_kwargs: Additional model parameters
            
        Returns:
            Trained model
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        # Create model
        input_dim = X_train.shape[1]
        model = LitPIC50(input_dim=input_dim, **model_kwargs)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            mode='min'
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename=f'{model_name}_best',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        
        # Logger
        logger = TensorBoardLogger('lightning_logs', name=model_name)
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            accelerator='auto',
            devices='auto'
        )
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Load best model
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            model = LitPIC50.load_from_checkpoint(best_model_path)
            self.logger.info(f"Loaded best model from {best_model_path}")
        
        return model
    
    def predict(self, model: LitPIC50, X: np.ndarray) -> np.ndarray:
        """Make predictions with a trained model.
        
        Args:
            model: Trained model
            X: Input features
            
        Returns:
            Predictions
        """
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def evaluate(self, model: LitPIC50, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained model.
        
        Args:
            model: Trained model
            X: Input features
            y: True targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(model, X)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        } 