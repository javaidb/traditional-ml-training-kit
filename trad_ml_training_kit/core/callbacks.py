"""Training callbacks for model training."""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "minimize"
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in monitored value to qualify as an improvement
            mode: Either "minimize" or "maximize"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = np.inf if mode == "minimize" else -np.inf
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
        
    def __call__(self, current_value: float, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            current_value: Current value of monitored metric
            epoch: Current epoch number
            
        Returns:
            bool: True if training should stop
        """
        if self.mode == "minimize":
            improvement = self.best_value - current_value > self.min_delta
        else:
            improvement = current_value - self.best_value > self.min_delta
            
        if improvement:
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
            logger.info(f"New best value: {current_value:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            logger.debug(
                f"No improvement for {self.counter} epochs. "
                f"Best value: {self.best_value:.6f} at epoch {self.best_epoch}"
            )
            
        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs. "
                f"Best value: {self.best_value:.6f} at epoch {self.best_epoch}"
            )
            self.should_stop = True
            
        return self.should_stop

class CallbackManager:
    """Manages multiple training callbacks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize callback manager.
        
        Args:
            config: Configuration dictionary
        """
        self.callbacks: List[Any] = []
        self._initialize_callbacks(config)
        
    def _initialize_callbacks(self, config: Dict[str, Any]) -> None:
        """Initialize callbacks based on configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Add early stopping if enabled
        if config.get('model', {}).get('early_stopping', {}).get('enabled', False):
            early_stopping_config = config['model']['early_stopping']
            optimization_direction = config.get('trainer', {}).get('optimization', {}).get('direction', 'minimize')
            
            self.callbacks.append(
                EarlyStopping(
                    patience=early_stopping_config['patience'],
                    min_delta=early_stopping_config['min_delta'],
                    mode=optimization_direction
                )
            )
            logger.info(
                f"Initialized early stopping with patience={early_stopping_config['patience']}, "
                f"min_delta={early_stopping_config['min_delta']}, mode={optimization_direction}"
            )
    
    def on_epoch_end(self, metrics: Dict[str, float], epoch: int) -> bool:
        """Called at the end of each epoch.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Current epoch number
            
        Returns:
            bool: True if training should stop
        """
        should_stop = False
        
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                # Use the primary loss function value for early stopping
                if callback(metrics['loss'], epoch):
                    should_stop = True
                    
        return should_stop 