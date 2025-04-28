"""Configuration handling for ML training."""

import yaml
from typing import Dict, Any, Optional
import logging
import numpy as np
import random
import torch
import os

logger = logging.getLogger(__name__)

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def validate_optimization_config(config: Dict[str, Any]) -> None:
    """Validate optimization-related configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'optimization' in config.get('trainer', {}):
        direction = config['trainer']['optimization'].get('direction')
        if direction not in ['minimize', 'maximize']:
            raise ValueError(
                f"Invalid optimization direction '{direction}'. "
                "Must be either 'minimize' or 'maximize'"
            )

def validate_early_stopping_config(config: Dict[str, Any]) -> None:
    """Validate early stopping configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'early_stopping' in config.get('model', {}):
        early_stopping = config['model']['early_stopping']
        if early_stopping.get('enabled', False):
            patience = early_stopping.get('patience')
            min_delta = early_stopping.get('min_delta')
            
            if not isinstance(patience, int) or patience <= 0:
                raise ValueError("early_stopping.patience must be a positive integer")
            if not isinstance(min_delta, (int, float)) or min_delta < 0:
                raise ValueError("early_stopping.min_delta must be a non-negative number")

def validate_experiment_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'tags' in config.get('experiment', {}):
        tags = config['experiment']['tags']
        if not isinstance(tags, dict):
            raise ValueError("experiment.tags must be a dictionary")
        
    if 'random_seed' in config.get('experiment', {}):
        seed = config['experiment']['random_seed']
        if not isinstance(seed, int):
            raise ValueError("experiment.random_seed must be an integer")

def apply_config(config: Dict[str, Any]) -> None:
    """Apply configuration settings to the environment.
    
    Args:
        config: Configuration dictionary
    """
    # Set random seed if specified
    if 'random_seed' in config.get('experiment', {}):
        set_random_seed(config['experiment']['random_seed'])
        logger.info(f"Set random seed to {config['experiment']['random_seed']}")

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict[str, Any]: Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate all sections
    validate_optimization_config(config)
    validate_early_stopping_config(config)
    validate_experiment_config(config)
    
    # Apply configuration
    apply_config(config)
    
    return config 