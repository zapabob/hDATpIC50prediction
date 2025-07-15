"""
Prediction script for pIC50 prediction models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .features.featurizer import MolecularFeaturizer
from .models.transformer import LitPIC50
from .utils.config import config


def setup_logging(log_file: str = 'predict.log') -> None:
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


def load_model(model_path: str) -> LitPIC50:
    """Load a trained model.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = LitPIC50.load_from_checkpoint(model_path)
        model.eval()
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_smiles_from_file(smiles_file: str) -> List[str]:
    """Load SMILES strings from a file.
    
    Args:
        smiles_file: Path to SMILES file
        
    Returns:
        List of SMILES strings
    """
    logger = logging.getLogger(__name__)
    
    if not Path(smiles_file).exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")
    
    try:
        with open(smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(smiles_list)} SMILES from {smiles_file}")
        return smiles_list
    except Exception as e:
        logger.error(f"Failed to load SMILES file: {e}")
        raise


def predict_single(model: LitPIC50, featurizer: MolecularFeaturizer, smiles: str) -> Optional[float]:
    """Predict pIC50 for a single SMILES string.
    
    Args:
        model: Trained model
        featurizer: Molecular featurizer
        smiles: SMILES string
        
    Returns:
        Predicted pIC50 or None if prediction fails
    """
    try:
        # Calculate features
        features = featurizer.calculate_features(smiles)
        if features is None:
            return None
        
        # Remove correlated features if needed
        if featurizer.feature_indices is not None:
            features = features[featurizer.feature_indices]
        
        # Scale features
        features_scaled, _ = featurizer.scale_features(features.reshape(1, -1))
        
        # Make prediction
        with torch.no_grad():
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            prediction = model(features_tensor)
            return float(prediction.item())
            
    except Exception as e:
        logging.error(f"Prediction failed for {smiles}: {e}")
        return None


def predict_batch(
    model: LitPIC50,
    featurizer: MolecularFeaturizer,
    smiles_list: List[str]
) -> List[Dict[str, Optional[float]]]:
    """Predict pIC50 for a batch of SMILES strings.
    
    Args:
        model: Trained model
        featurizer: Molecular featurizer
        smiles_list: List of SMILES strings
        
    Returns:
        List of prediction results
    """
    logger = logging.getLogger(__name__)
    results = []
    
    for i, smiles in enumerate(smiles_list):
        logger.info(f"Predicting {i+1}/{len(smiles_list)}: {smiles}")
        
        prediction = predict_single(model, featurizer, smiles)
        
        results.append({
            'smiles': smiles,
            'predicted_pIC50': prediction,
            'status': 'success' if prediction is not None else 'failed'
        })
    
    return results


def save_predictions(results: List[Dict], output_file: str) -> None:
    """Save prediction results to CSV file.
    
    Args:
        results: List of prediction results
        output_file: Output file path
    """
    logger = logging.getLogger(__name__)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(results)} predictions to {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    logger.info(f"Prediction summary: {successful} successful, {failed} failed")
    
    if successful > 0:
        predictions = [r['predicted_pIC50'] for r in results if r['predicted_pIC50'] is not None]
        logger.info(f"pIC50 range: {min(predictions):.2f} - {max(predictions):.2f}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict pIC50 values')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--smiles-file', help='Path to SMILES file (one per line)')
    parser.add_argument('--smiles', help='Single SMILES string')
    parser.add_argument('--out', required=True, help='Output CSV file path')
    parser.add_argument('--log-file', default='predict.log', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not args.smiles_file and not args.smiles:
        logger.error("Either --smiles-file or --smiles must be provided")
        sys.exit(1)
    
    if args.smiles_file and args.smiles:
        logger.error("Only one of --smiles-file or --smiles can be provided")
        sys.exit(1)
    
    try:
        # Load model
        model = load_model(args.model)
        
        # Initialize featurizer
        featurizer = MolecularFeaturizer()
        
        # Load SMILES
        if args.smiles_file:
            smiles_list = load_smiles_from_file(args.smiles_file)
        else:
            smiles_list = [args.smiles]
        
        # Make predictions
        results = predict_batch(model, featurizer, smiles_list)
        
        # Save results
        save_predictions(results, args.out)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 