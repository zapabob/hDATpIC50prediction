"""
Configuration classes for the hDAT pIC50 prediction system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for model training and prediction."""
    
    # Data parameters
    TEST_SIZE: float = 0.2
    RANDOM_SEED: int = 42
    CACHE_DIR: str = '.cache'
    DATA_DIR: str = 'data'
    MODEL_DIR: str = 'models'
    LOG_FILE: str = 'pic50_predictor.log'
    
    # Training parameters
    N_EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    EARLY_STOPPING: bool = True
    PATIENCE: int = 10
    SCHEDULER: bool = True
    
    # Model architecture
    TRANSFORMER_LAYERS: int = 2
    TRANSFORMER_HEADS: int = 4
    TRANSFORMER_DIM: int = 256
    DROPOUT: float = 0.1
    
    # Feature engineering
    ECFP4_RADIUS: int = 2
    ECFP4_BITS: int = 1024
    MACCS_BITS: int = 167
    CORRELATION_THRESHOLD: float = 0.9
    
    # Validation
    VALIDATION_SIZE: float = 0.2  # 80/20 split within training set


@dataclass
class TargetConfig:
    """Configuration for different molecular targets."""
    
    TARGETS: Optional[Dict[str, Tuple[str, str]]] = None
    
    def __post_init__(self):
        if self.TARGETS is None:
            self.TARGETS = {
                'CHEMBL238': ('DAT', 'Dopamine Transporter'),
                'CHEMBL224': ('5HT2A', '5-Hydroxytryptamine 2A Receptor'),
                'CHEMBL218': ('CB1', 'Cannabinoid Receptor 1'),
                'CHEMBL253': ('CB2', 'Cannabinoid Receptor 2'),
                'CHEMBL233': ('μ-opioid', 'μ-Opioid Receptor'),
                'CHEMBL236': ('δ-opioid', 'δ-Opioid Receptor'),
                'CHEMBL237': ('κ-opioid', 'κ-Opioid Receptor'),
            }
    
    def get_target_name(self, target_id: str) -> str:
        """Get the pretty name for a target ID."""
        if self.TARGETS is None:
            return target_id
        return self.TARGETS.get(target_id, (target_id, target_id))[0]
    
    def get_target_description(self, target_id: str) -> str:
        """Get the description for a target ID."""
        if self.TARGETS is None:
            return target_id
        return self.TARGETS.get(target_id, (target_id, target_id))[1]
    
    def list_targets(self) -> List[Tuple[str, str, str]]:
        """List all available targets with ID, name, and description."""
        if self.TARGETS is None:
            return []
        return [(tid, name, desc) for tid, (name, desc) in self.TARGETS.items()]


# Global configuration instances
config = ModelConfig()
target_config = TargetConfig() 