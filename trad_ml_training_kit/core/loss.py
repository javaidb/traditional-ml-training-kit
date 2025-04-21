"""Loss functions for model training and evaluation."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Callable, Dict, Union, List
import logging

logger = logging.getLogger(__name__)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error loss function.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error loss function.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        float: MAE value
    """
    return mean_absolute_error(y_true, y_pred)

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error loss function.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        float: MAPE value as percentage
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination) metric.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        float: R² value
    """
    return r2_score(y_true, y_pred)

# Dictionary mapping loss function names to their implementations
LOSS_FUNCTIONS: Dict[str, Callable] = {
    'rmse': rmse,
    'mae': mae,
    'mape': mape,
    'r2': r2
}

def get_loss_function(loss_name: str) -> Callable:
    """Get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        Callable: The requested loss function
        
    Raises:
        ValueError: If loss function is not supported
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Loss function '{loss_name}' not supported. "
            f"Available loss functions: {list(LOSS_FUNCTIONS.keys())}"
        )
    return LOSS_FUNCTIONS[loss_name]

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Union[str, List[str]] = ['rmse', 'mae', 'r2']
) -> Dict[str, float]:
    """Compute multiple metrics at once.
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        metrics: List of metric names or single metric name
        
    Returns:
        Dict[str, float]: Dictionary of metric names and their values
    """
    if isinstance(metrics, str):
        metrics = [metrics]
        
    results = {}
    for metric in metrics:
        try:
            loss_fn = get_loss_function(metric)
            results[metric] = loss_fn(y_true, y_pred)
        except ValueError as e:
            logger.warning(f"Skipping metric '{metric}': {str(e)}")
            
    return results 