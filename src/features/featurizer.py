"""
Molecular feature engineering for pIC50 prediction.
"""

import hashlib
import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys
from sklearn.preprocessing import RobustScaler

from ..utils.cache import FeatureCache
from ..utils.config import config


class MolecularFeaturizer:
    """Molecular feature engineering for pIC50 prediction."""
    
    def __init__(self, cache_dir: str = '.cache'):
        """Initialize the molecular featurizer.
        
        Args:
            cache_dir: Directory for caching features
        """
        self.cache = FeatureCache(cache_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize descriptor functions
        self._init_descriptor_functions()
        self._init_fingerprint_functions()
        self._init_smarts_patterns()
        
        # Feature names
        self.feature_names = self._get_feature_names()
        
        # Correlation pruning
        self.feature_indices = None
        self.removed_features = []
    
    def _init_descriptor_functions(self) -> None:
        """Initialize RDKit descriptor functions."""
        self.descriptor_functions = {
            'MolWt': Descriptors.MolWt,
            'MolLogP': Crippen.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'TPSA': Descriptors.TPSA,
            'FractionCSP3': Descriptors.FractionCSP3,
            'LabuteASA': Descriptors.LabuteASA,
            'BalabanJ': Descriptors.BalabanJ,
            'BertzCT': Descriptors.BertzCT,
        }
    
    def _init_fingerprint_functions(self) -> None:
        """Initialize fingerprint functions."""
        self.fingerprint_functions = {
            'ECFP4': partial(
                AllChem.GetMorganFingerprintAsBitVect,
                radius=config.ECFP4_RADIUS,
                nBits=config.ECFP4_BITS
            ),
            'MACCS': MACCSkeys.GenMACCSKeys
        }
    
    def _init_smarts_patterns(self) -> None:
        """Initialize SMARTS patterns for molecular flags."""
        self.smarts_patterns = {
            'HasIndole': Chem.MolFromSmarts('c1cc2c(cc1)[nH]c2'),
            'HasTryptamine': Chem.MolFromSmarts('CCN(CC)CCC1=CNC2=CC=CC=C12'),
            'HasPhenethylamine': Chem.MolFromSmarts('NCCc1ccc(O)cc1'),
            'MethoxyCount': Chem.MolFromSmarts('CO'),
            'HalogenCount': Chem.MolFromSmarts('[F,Cl,Br,I]'),
            'HasNNDimethyl': Chem.MolFromSmarts('N(C)C'),
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get all feature names.
        
        Returns:
            List of feature names
        """
        # Descriptor names
        descriptor_names = list(self.descriptor_functions.keys())
        
        # Fingerprint names
        fingerprint_names = []
        for name in self.fingerprint_functions.keys():
            if name == 'ECFP4':
                n_bits = config.ECFP4_BITS
            elif name == 'MACCS':
                n_bits = config.MACCS_BITS
            else:
                n_bits = 0
            fingerprint_names.extend([f"{name}_{i}" for i in range(n_bits)])
        
        # SMARTS pattern names
        smarts_names = list(self.smarts_patterns.keys())
        
        return descriptor_names + fingerprint_names + smarts_names
    
    def _get_smiles_hash(self, smiles: str) -> str:
        """Get MD5 hash of SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            MD5 hash string
        """
        return hashlib.md5(smiles.encode()).hexdigest()
    
    def calculate_features(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate molecular features for a SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature array or None if calculation fails
        """
        # Check cache first
        smiles_hash = self._get_smiles_hash(smiles)
        cached_features = self.cache.get(smiles_hash)
        if cached_features is not None:
            return cached_features
        
        # Calculate features
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.logger.warning(f"Invalid SMILES: {smiles}")
            return None
        
        try:
            features = self._calculate_molecular_features(mol)
            if features is not None:
                self.cache.save(smiles_hash, features)
            return features
        except Exception as e:
            self.logger.error(f"Feature calculation error for {smiles}: {e}")
            return None
    
    def _calculate_molecular_features(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """Calculate molecular features for an RDKit molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Feature array or None if calculation fails
        """
        try:
            # Calculate descriptors
            descriptors = []
            for func in self.descriptor_functions.values():
                try:
                    value = func(mol)
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    descriptors.append(float(value))
                except Exception:
                    descriptors.append(0.0)
            
            # Calculate fingerprints
            fingerprints = []
            for name, func in self.fingerprint_functions.items():
                try:
                    fp = func(mol)
                    if hasattr(fp, 'ToBitString'):
                        # Morgan fingerprint
                        bit_string = fp.ToBitString()
                        fingerprints.extend([int(b) for b in bit_string])
                    else:
                        # MACCS keys
                        bit_string = fp.ToBitString()
                        fingerprints.extend([int(b) for b in bit_string])
                except Exception:
                    # Use zeros for failed fingerprints
                    if name == 'ECFP4':
                        fingerprints.extend([0] * config.ECFP4_BITS)
                    elif name == 'MACCS':
                        fingerprints.extend([0] * config.MACCS_BITS)
            
            # Calculate SMARTS flags
            smarts_features = []
            for pattern_name, pattern in self.smarts_patterns.items():
                try:
                    if pattern_name in ['MethoxyCount', 'HalogenCount']:
                        # Count patterns
                        count = len(mol.GetSubstructMatches(pattern))
                        smarts_features.append(count)
                    else:
                        # Boolean flags
                        has_pattern = mol.HasSubstructMatch(pattern)
                        smarts_features.append(int(has_pattern))
                except Exception:
                    smarts_features.append(0)
            
            # Combine all features
            all_features = descriptors + fingerprints + smarts_features
            return np.array(all_features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Molecular feature calculation error: {e}")
            return None
    
    def calculate_batch_features(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
        """Calculate features for a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (feature_matrix, valid_indices)
        """
        features_list = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            features = self.calculate_features(smiles)
            if features is not None:
                features_list.append(features)
                valid_indices.append(i)
            else:
                self.logger.warning(f"Failed to calculate features for SMILES {i}: {smiles}")
        
        if not features_list:
            raise ValueError("No valid features could be calculated")
        
        feature_matrix = np.vstack(features_list)
        return feature_matrix, valid_indices
    
    def remove_correlated_features(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Remove highly correlated features.
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold (default from config)
            
        Returns:
            Feature matrix with correlated features removed
        """
        if threshold is None:
            threshold = config.CORRELATION_THRESHOLD
        
        # Calculate correlation matrix
        df = pd.DataFrame(X, columns=self.feature_names)
        corr_matrix = df.corr().abs()
        
        # Find highly correlated features
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            self.logger.info(f"Removing {len(to_drop)} highly correlated features")
            self.removed_features = to_drop.copy()
            
            # Remove features
            df_reduced = df.drop(columns=to_drop)
            X_reduced = df_reduced.values
            
            # Update feature names and indices
            self.feature_names = df_reduced.columns.tolist()
            self.feature_indices = [
                i for i, name in enumerate(self._get_feature_names())
                if name in self.feature_names
            ]
            
            return X_reduced
        else:
            self.logger.info("No highly correlated features found")
            return X
    
    def get_feature_names(self) -> List[str]:
        """Get current feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
    
    def get_feature_summary(self, X: np.ndarray) -> Dict[str, float]:
        """Get summary statistics for features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with feature statistics
        """
        return {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_mean': np.mean(X, axis=0).tolist(),
            'feature_std': np.std(X, axis=0).tolist(),
            'feature_min': np.min(X, axis=0).tolist(),
            'feature_max': np.max(X, axis=0).tolist(),
        }
    
    def scale_features(self, X: np.ndarray, scaler: Optional[RobustScaler] = None) -> Tuple[np.ndarray, RobustScaler]:
        """Scale features using RobustScaler.
        
        Args:
            X: Feature matrix
            scaler: Pre-fitted scaler (if None, fit new one)
            
        Returns:
            Tuple of (scaled_features, fitted_scaler)
        """
        if scaler is None:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        return X_scaled, scaler 