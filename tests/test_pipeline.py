"""
Tests for data pipeline components.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import ChEMBLDataLoader
from src.features.featurizer import MolecularFeaturizer
from src.utils.config import config


class TestChEMBLDataLoader:
    """Test ChEMBL data loader."""
    
    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        loader = ChEMBLDataLoader()
        assert loader.cache is not None
    
    def test_data_preprocessing(self):
        """Test data preprocessing with sample data."""
        # Create sample data
        sample_data = pd.DataFrame({
            'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'canonical_smiles': [
                'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)S(=O)(=O)N)C(=O)O'  # Sulfamethoxazole
            ],
            'standard_value': [1000, 5000, 2500]  # nM
        })
        
        # Test preprocessing
        result = ChEMBLDataLoader._preprocess_data(self, sample_data)
        
        assert 'pIC50' in result.columns
        assert len(result) == 3
        assert all(result['pIC50'] > 0)
        assert all(result['pIC50'] < 15)  # Reasonable range
    
    def test_data_splitting(self):
        """Test data splitting methods."""
        loader = ChEMBLDataLoader()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'canonical_smiles': [
                'CC(=O)OC1=CC=CC=C1C(=O)O',
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
                'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)S(=O)(=O)N)C(=O)O',
                'CC1=CC=C(C=C1)C2=CC=CC=C2',
                'CC1=CC=C(C=C1)C2=CC=CC=C2C(=O)O'
            ],
            'pIC50': [5.0, 4.5, 6.2, 3.8, 5.5]
        })
        
        # Test random split
        train_df, test_df = loader.split_data(sample_data, split_method='random')
        assert len(train_df) + len(test_df) == len(sample_data)
        
        # Test scaffold split
        train_df, test_df = loader.split_data(sample_data, split_method='scaffold')
        assert len(train_df) + len(test_df) == len(sample_data)


class TestMolecularFeaturizer:
    """Test molecular featurizer."""
    
    def test_featurizer_initialization(self):
        """Test featurizer initialization."""
        featurizer = MolecularFeaturizer()
        assert featurizer.descriptor_functions is not None
        assert featurizer.fingerprint_functions is not None
        assert featurizer.smarts_patterns is not None
    
    def test_feature_calculation(self):
        """Test feature calculation for valid SMILES."""
        featurizer = MolecularFeaturizer()
        
        # Test with aspirin
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        features = featurizer.calculate_features(smiles)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        
        # Check feature dimensions
        expected_features = (
            len(featurizer.descriptor_functions) +
            config.ECFP4_BITS +
            config.MACCS_BITS +
            len(featurizer.smarts_patterns)
        )
        assert len(features) == expected_features
    
    def test_invalid_smiles(self):
        """Test feature calculation for invalid SMILES."""
        featurizer = MolecularFeaturizer()
        
        # Test with invalid SMILES
        invalid_smiles = 'invalid_smiles'
        features = featurizer.calculate_features(invalid_smiles)
        
        assert features is None
    
    def test_batch_feature_calculation(self):
        """Test batch feature calculation."""
        featurizer = MolecularFeaturizer()
        
        smiles_list = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)S(=O)(=O)N)C(=O)O'  # Sulfamethoxazole
        ]
        
        feature_matrix, valid_indices = featurizer.calculate_batch_features(smiles_list)
        
        assert isinstance(feature_matrix, np.ndarray)
        assert len(valid_indices) == len(smiles_list)
        assert feature_matrix.shape[0] == len(smiles_list)
    
    def test_correlation_pruning(self):
        """Test correlation-based feature pruning."""
        featurizer = MolecularFeaturizer()
        
        # Create sample feature matrix with correlated features
        n_samples = 100
        n_features = 10
        
        # Create correlated features
        base_feature = np.random.randn(n_samples)
        X = np.column_stack([
            base_feature,
            base_feature + 0.01 * np.random.randn(n_samples),  # Highly correlated
            np.random.randn(n_samples),  # Uncorrelated
            base_feature + 0.02 * np.random.randn(n_samples),  # Highly correlated
            np.random.randn(n_samples),  # Uncorrelated
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples)
        ])
        
        # Update feature names for testing
        featurizer.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Apply correlation pruning
        X_pruned = featurizer.remove_correlated_features(X, threshold=0.9)
        
        # Should remove some correlated features
        assert X_pruned.shape[1] < X.shape[1]
        assert X_pruned.shape[0] == X.shape[0]
    
    def test_feature_scaling(self):
        """Test feature scaling."""
        featurizer = MolecularFeaturizer()
        
        # Create sample data
        X = np.random.randn(100, 10)
        
        # Scale features
        X_scaled, scaler = featurizer.scale_features(X)
        
        assert X_scaled.shape == X.shape
        assert scaler is not None
        
        # Test with pre-fitted scaler
        X_new = np.random.randn(50, 10)
        X_new_scaled, _ = featurizer.scale_features(X_new, scaler)
        
        assert X_new_scaled.shape == X_new.shape


class TestDataIntegration:
    """Test data integration pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete data processing pipeline."""
        # This test would require actual ChEMBL data
        # For now, we'll test the components separately
        pass
    
    def test_feature_cache(self):
        """Test feature caching functionality."""
        featurizer = MolecularFeaturizer()
        
        # Test caching
        smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
        
        # First calculation
        features1 = featurizer.calculate_features(smiles)
        assert features1 is not None
        
        # Second calculation (should use cache)
        features2 = featurizer.calculate_features(smiles)
        assert features2 is not None
        
        # Should be identical
        np.testing.assert_array_equal(features1, features2)


if __name__ == '__main__':
    pytest.main([__file__]) 