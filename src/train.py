"""
Training script for pIC50 prediction models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data.loader import ChEMBLDataLoader
from .features.featurizer import MolecularFeaturizer
from .models.transformer import LitPIC50, PIC50Trainer
from .utils.config import config, target_config


def setup_logging(log_file: str = 'train.log') -> None:
    """Setup logging configuration.
    
    Args:
        log_file: Log file path
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_and_preprocess_data(
    target_id: str,
    split_method: str = 'random',
    force_refresh: bool = False
) -> tuple:
    """Load and preprocess data for training.
    
    Args:
        target_id: ChEMBL target ID
        split_method: Data split method ('random' or 'scaffold')
        force_refresh: Force refresh from ChEMBL
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, featurizer)
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    data_loader = ChEMBLDataLoader()
    df = data_loader.load_chembl(target_id, force_refresh=force_refresh)
    
    # Split data
    train_df, test_df = data_loader.split_data(df, split_method=split_method)
    
    # Further split training data for validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_SEED
    )
    
    logger.info(f"Data split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Calculate features
    featurizer = MolecularFeaturizer()
    
    # Train features
    X_train, train_valid_indices = featurizer.calculate_batch_features(
        train_df['canonical_smiles'].tolist()
    )
    y_train = train_df['pIC50'].values[train_valid_indices]
    
    # Validation features
    X_val, val_valid_indices = featurizer.calculate_batch_features(
        val_df['canonical_smiles'].tolist()
    )
    y_val = val_df['pIC50'].values[val_valid_indices]
    
    # Test features
    X_test, test_valid_indices = featurizer.calculate_batch_features(
        test_df['canonical_smiles'].tolist()
    )
    y_test = test_df['pIC50'].values[test_valid_indices]
    
    # Remove correlated features (using training data only)
    X_train = featurizer.remove_correlated_features(X_train)
    X_val = featurizer.remove_correlated_features(X_val)
    X_test = featurizer.remove_correlated_features(X_test)
    
    # Scale features
    X_train, scaler = featurizer.scale_features(X_train)
    X_val, _ = featurizer.scale_features(X_val, scaler)
    X_test, _ = featurizer.scale_features(X_test, scaler)
    
    logger.info(f"Feature matrix shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, featurizer


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str
) -> float:
    """Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_name: Model name
        
    Returns:
        Validation RMSE (to be minimized)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
    
    # Create trainer
    trainer = PIC50Trainer(
        max_epochs=50,  # Shorter training for optimization
        patience=10,
        batch_size=batch_size
    )
    
    try:
        # Train model
        model = trainer.train(
            X_train, y_train, X_val, y_val, model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Evaluate on validation set
        metrics = trainer.evaluate(model, X_val, y_val)
        return metrics['rmse']  # Minimize RMSE
        
    except Exception as e:
        logging.error(f"Trial failed: {e}")
        return float('inf')  # Return high value for failed trials


def train_model(
    target_id: str,
    split_method: str = 'random',
    use_optuna: bool = False,
    n_trials: int = 20,
    big_model: bool = False,
    force_refresh: bool = False
) -> Dict:
    """Train a pIC50 prediction model.
    
    Args:
        target_id: ChEMBL target ID
        split_method: Data split method
        use_optuna: Whether to use Optuna for hyperparameter optimization
        n_trials: Number of Optuna trials
        big_model: Whether to use larger model architecture
        force_refresh: Force refresh from ChEMBL
        
    Returns:
        Dictionary with training results
    """
    logger = logging.getLogger(__name__)
    
    # Get target name
    target_name = target_config.get_target_name(target_id)
    model_name = f"{target_name}_{target_id}"
    
    logger.info(f"Training model for {target_name} ({target_id})")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, featurizer = load_and_preprocess_data(
        target_id, split_method, force_refresh
    )
    
    # Model parameters
    if big_model:
        model_params = {
            'num_layers': 4,
            'num_heads': 8,
            'dim_feedforward': 512,
            'dropout': 0.2
        }
    else:
        model_params = {
            'num_layers': 2,
            'num_heads': 4,
            'dim_feedforward': 256,
            'dropout': 0.1
        }
    
    if use_optuna:
        logger.info(f"Running Optuna optimization with {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, model_name),
            n_trials=n_trials
        )
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        
        # Save best parameters
        params_file = Path(config.MODEL_DIR) / f"{model_name}_best_params.json"
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Update model parameters with best values
        model_params.update(best_params)
    
    # Train final model
    trainer = PIC50Trainer(
        max_epochs=config.N_EPOCHS,
        patience=config.PATIENCE,
        batch_size=model_params.get('batch_size', config.BATCH_SIZE)
    )
    
    model = trainer.train(
        X_train, y_train, X_val, y_val, model_name,
        **model_params
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(model, X_test, y_test)
    val_metrics = trainer.evaluate(model, X_val, y_val)
    train_metrics = trainer.evaluate(model, X_train, y_train)
    
    # Save results
    results = {
        'target_id': target_id,
        'target_name': target_name,
        'split_method': split_method,
        'use_optuna': use_optuna,
        'n_trials': n_trials if use_optuna else 0,
        'big_model': big_model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_params': model_params,
        'feature_names': featurizer.get_feature_names(),
        'n_features': len(featurizer.get_feature_names()),
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test)
    }
    
    # Save results
    results_file = Path(config.MODEL_DIR) / f"{model_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training completed. Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train pIC50 prediction model')
    parser.add_argument('--target', required=True, help='ChEMBL target ID')
    parser.add_argument('--split', default='random', choices=['random', 'scaffold'],
                       help='Data split method')
    parser.add_argument('--optuna', type=int, help='Number of Optuna trials')
    parser.add_argument('--big-model', action='store_true', help='Use larger model architecture')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh from ChEMBL')
    parser.add_argument('--log-file', default='train.log', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Validate target ID
    if args.target not in target_config.TARGETS:
        logger.error(f"Invalid target ID: {args.target}")
        logger.info(f"Available targets: {list(target_config.TARGETS.keys())}")
        sys.exit(1)
    
    # Train model
    try:
        results = train_model(
            target_id=args.target,
            split_method=args.split,
            use_optuna=args.optuna is not None,
            n_trials=args.optuna or 20,
            big_model=args.big_model,
            force_refresh=args.force_refresh
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to {config.MODEL_DIR}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 