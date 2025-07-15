"""
hDAT pIC50 Prediction Package

A comprehensive molecular property prediction system supporting multiple targets:
- DAT (CHEMBL238)
- 5HT2A (CHEMBL224) 
- CB1 (CHEMBL218)
- CB2 (CHEMBL253)
- μ-opioid (CHEMBL233)
- δ-opioid (CHEMBL236)
- κ-opioid (CHEMBL237)
"""

__version__ = "2.0.0"
__author__ = "zapabob"
__email__ = "zapabob@example.com"

from .data.loader import ChEMBLDataLoader
from .features.featurizer import MolecularFeaturizer
from .models.transformer import LitPIC50
from .utils.config import ModelConfig
from .utils.cache import FeatureCache

__all__ = [
    "ChEMBLDataLoader",
    "MolecularFeaturizer", 
    "LitPIC50",
    "ModelConfig",
    "FeatureCache",
] 