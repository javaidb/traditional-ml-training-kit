import os
import logging
import yaml
from typing import Dict, Any, Optional
import mlflow
from .training import ModelTrainer

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(level=level)
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep update a configuration dictionary with new values.
    
    Args:
        base_config: Base configuration dictionary
        updates: Dictionary of updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    config = base_config.copy()
    
    def _update_dict(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict(d[k].copy(), v)
            else:
                d[k] = v
        return d
    
    return _update_dict(config, updates)

def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> None:
    """Set up MLflow tracking."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

def setup_training(
    config_path: str,
    mlflow_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None
) -> ModelTrainer:
    """
    Set up training environment and return trainer.
    
    Args:
        config_path: Path to configuration file
        mlflow_uri: Optional MLflow tracking URI
        experiment_name: Optional experiment name override
        config_override: Optional dictionary to override config values
        
    Returns:
        Configured ModelTrainer instance
    """
    # Set up logging
    logger = setup_logging()
    
    # Load and update configuration
    config = load_config(config_path)
    if config_override:
        # Deep update the config with overrides
        config = update_config(config, config_override)
        logger.info("Applied configuration overrides")
    
    # Set up MLflow
    setup_mlflow(
        tracking_uri=mlflow_uri,
        experiment_name=experiment_name or config['experiment']['name']
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        config=config,
        experiment_name=experiment_name
    )
    
    return trainer 