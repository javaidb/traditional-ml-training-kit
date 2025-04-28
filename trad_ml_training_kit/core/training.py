import mlflow
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from .models import get_model_by_name
from .data import DataModule
from .logging import MLflowLogger
from datetime import datetime
import pytz
import shutil
import os
from pathlib import Path
import requests
import time

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer class for traditional ML models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None
    ):
        """Initialize trainer."""
        self.config = config
        self.experiment_name = experiment_name or config['experiment']['name']
        self.logger = logging.getLogger(__name__)
        
        # Verify MLflow connection
        self._verify_mlflow_connection()
        
        # Initialize MLflow logger
        self.mlflow_logger = MLflowLogger(
            experiment_name=self.experiment_name,
            experiment_tags=config.get('experiment', {}).get('tags', {}),
            model_artifact_path=config.get('model', {}).get('artifact_path', 'model')
        )
    
    def _verify_mlflow_connection(self, max_retries: int = 5) -> None:
        """Verify connection to MLflow server.
        
        Args:
            max_retries: Maximum number of connection attempts
        
        Raises:
            ConnectionError: If unable to connect to MLflow server
        """
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.logger.info(f"Verifying connection to MLflow server at {mlflow_uri}")
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{mlflow_uri}/health")
                if response.status_code == 200:
                    self.logger.info("Successfully connected to MLflow server")
                    return
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {i+1}/{max_retries} failed: {str(e)}")
                if i < max_retries - 1:
                    time.sleep(2 ** i)  # Exponential backoff
        
        raise ConnectionError(f"Failed to connect to MLflow server at {mlflow_uri}")
    
    def _get_metric_functions(self) -> Dict[str, Callable]:
        """Get metric functions based on task type."""
        if self.config['experiment']['task'] == 'regression':
            return {
                'mse': mean_squared_error,
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error,
                'r2': r2_score,
                'mape': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
        else:
            raise ValueError(f"Unsupported task type: {self.config['experiment']['task']}")
    
    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            prefix: Metric name prefix
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X)
        metrics = {}
        
        for name, func in self._get_metric_functions().items():
            metric_value = func(y, y_pred)
            metrics[f'{prefix}{name}'] = metric_value
        
        return metrics
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Get hyperparameter suggestions based on model type
        model_type = self.config['model']['type']
        if model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0)
            }
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
            }
        else:
            raise ValueError(f"Unsupported model type for optimization: {model_type}")
        
        # Create and train model
        model = get_model_by_name(model_type, params)
        X_train, y_train = self.data_module.get_train_data()
        X_val, y_val = self.data_module.get_val_data()
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        metrics = self._evaluate_model(model, X_val, y_val)
        
        # Log trial metrics
        self.mlflow_logger.log_metrics(metrics, step=trial.number)
        
        # Return the metric to optimize
        return metrics['rmse']  # Assuming we want to minimize RMSE

    def train(self) -> Any:
        """Train the model with hyperparameter optimization."""
        # Set up data module
        self.data_module = DataModule(self.config)
        
        # Prepare data
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_names
        ) = self.data_module.prepare_data()
        
        # Start MLflow run using the logger
        with self.mlflow_logger:
            self.logger.info(f"Started training run")
            
            # Log data config
            self.mlflow_logger.log_params({
                'data_path': self.config['data_module']['csv_path'],
                'target_column': self.config['data_module']['target_column'],
                'train_ratio': self.config['data_module'].get('validation', {}).get('train_ratio', 0.7),
                'val_ratio': self.config['data_module'].get('validation', {}).get('val_ratio', 0.15),
                'test_ratio': self.config['data_module'].get('validation', {}).get('test_ratio', 0.15),
                'n_features': len(feature_names),
                'n_samples': len(X_train) + len(X_val) + len(X_test)
            })
            
            # Log feature names
            self.mlflow_logger.log_params({'feature_names': ','.join(feature_names)})
            
            # Log data distribution plots
            try:
                dist_dir = self.data_module.distribution_plots_dir
                if not os.path.exists(dist_dir):
                    raise FileNotFoundError(f"Distribution plots directory not found: {dist_dir}")
                
                self.logger.info(f"Logging data distribution plots from {dist_dir}")
                self.logger.debug(f"Directory contents before logging: {os.listdir(dist_dir)}")
                
                # Verify MLflow artifact logging is working
                test_file = os.path.join(dist_dir, "test.txt")
                with open(test_file, "w") as f:
                    f.write("Test artifact logging")
                
                mlflow.log_artifact(test_file, "data_distributions")
                self.logger.info("Test artifact logging successful")
                
                # Log all distribution plots
                mlflow.log_artifacts(dist_dir, "data_distributions")
                self.logger.info("Successfully logged data distribution plots to MLflow")
                
                # Clean up temporary directory
                shutil.rmtree(dist_dir)
                self.logger.info(f"Cleaned up temporary directory: {dist_dir}")
            except Exception as e:
                self.logger.error(f"Failed to log data distribution plots: {str(e)}")
                self.logger.error(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
                self.logger.error(f"Current working directory: {os.getcwd()}")
                raise
            
            # Hyperparameter optimization
            if self.config['trainer']['hyperparameter_tuning']['enabled']:
                study = optuna.create_study(direction='minimize')
                study.optimize(
                    self._objective,
                    n_trials=self.config['trainer']['hyperparameter_tuning']['n_trials']
                )
                
                # Get best parameters
                best_params = study.best_params
                self.logger.info(f"Best parameters: {best_params}")
                self.mlflow_logger.log_params(best_params)
                
                # Log optimization history
                self.mlflow_logger.log_optimization_history(study)
            else:
                best_params = self.config['model'].get('params', {})
            
            # Train final model with best parameters
            model = get_model_by_name(self.config['model']['type'], best_params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_metrics = self._evaluate_model(
                model,
                X_train,
                y_train,
                prefix='train_'
            )
            val_metrics = self._evaluate_model(
                model,
                X_val,
                y_val,
                prefix='val_'
            )
            test_metrics = self._evaluate_model(
                model,
                X_test,
                y_test,
                prefix='test_'
            )
            
            # Log all metrics
            all_metrics = {
                **train_metrics,
                **val_metrics,
                **test_metrics
            }
            self.mlflow_logger.log_metrics(all_metrics)
            
            # Log prediction plots
            self.mlflow_logger.log_prediction_plots(
                y_train,
                model.predict(X_train),
                'train'
            )
            self.mlflow_logger.log_prediction_plots(
                y_val,
                model.predict(X_val),
                'val'
            )
            self.mlflow_logger.log_prediction_plots(
                y_test,
                model.predict(X_test),
                'test'
            )
            
            # Log feature importance
            if hasattr(model, 'get_feature_importance'):
                self.mlflow_logger.log_feature_importance(
                    model.get_feature_importance(),
                    feature_names
                )
            
            # Log model with signature
            try:
                self.mlflow_logger.log_model(
                    model=model.model,
                    model_name="model",
                    input_example=X_train.iloc[:5],
                    registered_model_name=f"{self.config['model']['type']}_{self.experiment_name}"
                )
            except Exception as e:
                self.logger.warning(f"Could not log model with signature: {str(e)}")
                self.mlflow_logger.log_model(model.model, "model")
            
            self.logger.info("Training completed successfully")
            self.logger.info(f"Test metrics: {test_metrics}")
            
            return model 