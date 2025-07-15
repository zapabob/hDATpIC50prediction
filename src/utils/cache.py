"""
Caching utilities for molecular features and data.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


class FeatureCache:
    """Molecular feature caching system."""
    
    def __init__(self, cache_dir: str = '.cache'):
        """Initialize the feature cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str, suffix: str = '.npz') -> Path:
        """Get cache file path for a given key.
        
        Args:
            key: Cache key (usually SMILES hash)
            suffix: File extension
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{key}{suffix}"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get cached features.
        
        Args:
            key: Cache key (usually SMILES hash)
            
        Returns:
            Cached features or None if not found
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                return np.load(cache_path)['features']
            except Exception:
                return None
        return None
    
    def save(self, key: str, features: np.ndarray) -> None:
        """Save features to cache.
        
        Args:
            key: Cache key (usually SMILES hash)
            features: Feature array to cache
        """
        cache_path = self._get_cache_path(key)
        np.savez_compressed(cache_path, features=features)
    
    def clear(self) -> None:
        """Clear all cached features."""
        if self.cache_dir.exists():
            for item in self.cache_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)


class DataCache:
    """Data caching system for ChEMBL data."""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize the data cache.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_data_path(self, target_id: str, suffix: str = '.parquet') -> Path:
        """Get data file path for a target.
        
        Args:
            target_id: ChEMBL target ID
            suffix: File extension
            
        Returns:
            Path to data file
        """
        target_dir = self.data_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"data{suffix}"
    
    def get(self, target_id: str) -> Optional[pd.DataFrame]:
        """Get cached data for a target.
        
        Args:
            target_id: ChEMBL target ID
            
        Returns:
            Cached DataFrame or None if not found
        """
        data_path = self._get_data_path(target_id)
        if data_path.exists():
            try:
                return pd.read_parquet(data_path)
            except Exception:
                return None
        return None
    
    def save(self, target_id: str, data: pd.DataFrame) -> None:
        """Save data to cache.
        
        Args:
            target_id: ChEMBL target ID
            data: DataFrame to cache
        """
        data_path = self._get_data_path(target_id)
        data.to_parquet(data_path, index=False)
    
    def get_raw_path(self, target_id: str) -> Path:
        """Get path for raw data dump.
        
        Args:
            target_id: ChEMBL target ID
            
        Returns:
            Path to raw data file
        """
        target_dir = self.data_dir / target_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / "raw_data.pkl"
    
    def save_raw(self, target_id: str, raw_data: Any) -> None:
        """Save raw data dump.
        
        Args:
            target_id: ChEMBL target ID
            raw_data: Raw data to save
        """
        raw_path = self.get_raw_path(target_id)
        with open(raw_path, 'wb') as f:
            pickle.dump(raw_data, f)
    
    def get_raw(self, target_id: str) -> Optional[Any]:
        """Get raw data dump.
        
        Args:
            target_id: ChEMBL target ID
            
        Returns:
            Raw data or None if not found
        """
        raw_path = self.get_raw_path(target_id)
        if raw_path.exists():
            try:
                with open(raw_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None 