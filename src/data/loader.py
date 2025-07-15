"""
Data loading and preprocessing for ChEMBL molecular data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from ..utils.cache import DataCache
from ..utils.config import config


class ScaffoldSplitter:
    """Scaffold-based data splitter for molecular data."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize scaffold splitter.
        
        Args:
            test_size: Fraction of data for test set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, smiles_list: List[str]) -> Tuple[List[int], List[int]]:
        """Split data based on molecular scaffolds.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (train_indices, test_indices)
        """
        # Generate scaffolds for each molecule
        scaffolds = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
            else:
                scaffolds.append('invalid')
        
        # Group molecules by scaffold
        scaffold_groups: Dict[str, List[int]] = {}
        for i, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_groups:
                scaffold_groups[scaffold] = []
            scaffold_groups[scaffold].append(i)
        
        # Split scaffolds into train/test
        scaffold_list = list(scaffold_groups.keys())
        train_scaffolds, test_scaffolds = train_test_split(
            scaffold_list,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Get indices for each split
        train_indices = []
        for scaffold in train_scaffolds:
            train_indices.extend(scaffold_groups[scaffold])
        
        test_indices = []
        for scaffold in test_scaffolds:
            test_indices.extend(scaffold_groups[scaffold])
        
        return train_indices, test_indices


class ChEMBLDataLoader:
    """Data loader for ChEMBL molecular data."""
    
    def __init__(self, cache_dir: str = 'data'):
        """Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache = DataCache(cache_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_chembl(self, target_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """Load data from ChEMBL for a specific target.
        
        Args:
            target_id: ChEMBL target ID (e.g., 'CHEMBL238')
            force_refresh: Force refresh from ChEMBL API
            
        Returns:
            DataFrame with canonical_smiles and pIC50 columns
            
        Raises:
            ValueError: If data cannot be loaded
        """
        # Check cache first
        if not force_refresh:
            cached_data = self.cache.get(target_id)
            if cached_data is not None:
                self.logger.info(f"Loaded {len(cached_data)} samples from cache for {target_id}")
                return cached_data
        
        # Fetch from ChEMBL
        self.logger.info(f"Fetching data from ChEMBL for target {target_id}")
        df = self._fetch_from_chembl(target_id)
        
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Cache processed data
        self.cache.save(target_id, df)
        
        self.logger.info(f"Loaded and cached {len(df)} samples for {target_id}")
        return df
    
    def _fetch_from_chembl(self, target_id: str) -> pd.DataFrame:
        """Fetch raw data from ChEMBL API.
        
        Args:
            target_id: ChEMBL target ID
            
        Returns:
            Raw DataFrame from ChEMBL
            
        Raises:
            ValueError: If data cannot be fetched
        """
        try:
            # Check if raw data is cached
            raw_data = self.cache.get_raw(target_id)
            if raw_data is not None:
                return pd.DataFrame(raw_data)
            
            # Fetch from ChEMBL API
            target = new_client.target
            activity = new_client.activity
            
            # Get target information
            target_info = target.filter(target_chembl_id=target_id)
            if not target_info:
                raise ValueError(f"Target {target_id} not found in ChEMBL")
            
            # Get activities
            activities = activity.filter(
                target_chembl_id=target_id,
                standard_type="IC50",
                standard_units="nM"
            )
            
            if not activities:
                raise ValueError(f"No IC50 data found for target {target_id}")
            
            # Convert to DataFrame
            df = pd.DataFrame(activities)
            
            # Cache raw data
            self.cache.save_raw(target_id, activities)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data from ChEMBL: {e}")
            raise ValueError(f"Failed to fetch data for {target_id}: {e}")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw ChEMBL data.
        
        Args:
            df: Raw DataFrame from ChEMBL
            
        Returns:
            Preprocessed DataFrame with canonical_smiles and pIC50 columns
        """
        # Select required columns
        required_cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter data
        df = df[required_cols].copy()
        
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna()
        if len(df) < initial_count:
            self.logger.warning(f"Removed {initial_count - len(df)} rows with missing values")
        
        # Convert standard_value to numeric
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df.dropna(subset=['standard_value'])
        
        # Filter by IC50 range (reasonable values)
        df = df[df['standard_value'] > 0]
        df = df[df['standard_value'] < 1_000_000]  # 1 mM max
        
        # Calculate pIC50
        df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
        
        # Filter by pIC50 range (reasonable values)
        df = df[df['pIC50'] >= 0]  # pIC50 should be positive
        df = df[df['pIC50'] <= 15]  # Reasonable upper bound
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['canonical_smiles'])
        
        # Validate SMILES
        valid_smiles = []
        valid_pic50 = []
        valid_ids = []
        
        for _, row in df.iterrows():
            smiles = row['canonical_smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
                valid_pic50.append(row['pIC50'])
                valid_ids.append(row['molecule_chembl_id'])
        
        result_df = pd.DataFrame({
            'molecule_chembl_id': valid_ids,
            'canonical_smiles': valid_smiles,
            'pIC50': valid_pic50
        })
        
        self.logger.info(f"Preprocessing complete: {len(result_df)} valid samples")
        return result_df
    
    def split_data(
        self,
        df: pd.DataFrame,
        split_method: str = 'random',
        test_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            split_method: 'random' or 'scaffold'
            test_size: Fraction for test set (default from config)
            random_state: Random seed (default from config)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if test_size is None:
            test_size = config.TEST_SIZE
        if random_state is None:
            random_state = config.RANDOM_SEED
        
        if split_method == 'random':
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state
            )
        elif split_method == 'scaffold':
            splitter = ScaffoldSplitter(test_size=test_size, random_state=random_state)
            train_indices, test_indices = splitter.split(df['canonical_smiles'].tolist())
            train_df = df.iloc[train_indices].reset_index(drop=True)
            test_df = df.iloc[test_indices].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        self.logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test samples")
        return train_df, test_df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get summary statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'n_samples': len(df),
            'pIC50_mean': df['pIC50'].mean(),
            'pIC50_std': df['pIC50'].std(),
            'pIC50_min': df['pIC50'].min(),
            'pIC50_max': df['pIC50'].max(),
            'unique_smiles': df['canonical_smiles'].nunique()
        } 