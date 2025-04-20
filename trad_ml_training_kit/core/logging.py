import mlflow
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class MLflowLogger:
    """MLflow experiment tracking wrapper."""
    
    def __init__(self, experiment_name: str):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log model parameters."""
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log model metrics."""
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    def log_model(self, model: Any, model_name: str) -> None:
        """Log the model artifact."""
        mlflow.sklearn.log_model(model, model_name)
    
    def log_feature_importance(self, feature_importance: pd.Series, artifact_path: str) -> None:
        """
        Log feature importance plot and values.
        
        Args:
            feature_importance: Series containing feature importance scores
            artifact_path: Path to save the feature importance plot
        """
        # Log feature importance values
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Log feature importance plot
        if artifact_path:
            mlflow.log_artifact(artifact_path)
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run() 