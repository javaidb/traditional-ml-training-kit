import mlflow
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from .models import get_model_by_name
from .data_handler import DataHandler

class ModelTrainer:
    """Trainer class for traditional ML models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            experiment_name: Optional MLflow experiment name override
        """
        self.config = config
        self.experiment_name = experiment_name or config['experiment']['name']
        self.logger = logging.getLogger(__name__)
    
    def _get_metric_functions(self) -> Dict[str, Callable]:
        """Get metric functions based on task type."""
        if self.config['experiment']['task'] == 'regression':
            return {
                'mse': mean_squared_error,
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error,
                'r2': r2_score
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
        model.fit(self.data_handler.X_train, self.data_handler.y_train)
        
        # Evaluate on validation set
        metrics = self._evaluate_model(
            model,
            self.data_handler.X_val,
            self.data_handler.y_val
        )
        
        # Return the metric to optimize
        return metrics['rmse']  # Assuming we want to minimize RMSE
    
    def train(self) -> Any:
        """Train the model with hyperparameter optimization."""
        # Set up data handler
        data_config = self.config['data_module']
        self.data_handler = DataHandler(
            csv_path=data_config['csv_path'],
            target_column=data_config['target_column'],
            categorical_columns=data_config.get('categorical_columns'),
            numerical_columns=data_config.get('numerical_columns'),
            train_ratio=data_config.get('train_ratio', 0.7),
            val_ratio=data_config.get('val_ratio', 0.15),
            test_ratio=data_config.get('test_ratio', 0.15)
        )
        self.data_handler.setup()
        
        # Start MLflow run
        with mlflow.start_run() as run:
            self.logger.info(f"Started MLflow run: {run.info.run_id}")
            
            # Log data config
            mlflow.log_params({
                'data_path': data_config['csv_path'],
                'target_column': data_config['target_column'],
                'train_ratio': data_config.get('train_ratio', 0.7),
                'val_ratio': data_config.get('val_ratio', 0.15),
                'test_ratio': data_config.get('test_ratio', 0.15)
            })
            
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
                mlflow.log_params(best_params)
            else:
                best_params = self.config['model'].get('params', {})
            
            # Train final model with best parameters
            model = get_model_by_name(self.config['model']['type'], best_params)
            model.fit(self.data_handler.X_train, self.data_handler.y_train)
            
            # Evaluate model
            train_metrics = self._evaluate_model(
                model,
                self.data_handler.X_train,
                self.data_handler.y_train,
                prefix='train_'
            )
            val_metrics = self._evaluate_model(
                model,
                self.data_handler.X_val,
                self.data_handler.y_val,
                prefix='val_'
            )
            test_metrics = self._evaluate_model(
                model,
                self.data_handler.X_test,
                self.data_handler.y_test,
                prefix='test_'
            )
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics, **test_metrics}
            mlflow.log_metrics(all_metrics)
            
            # Log feature importance
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                importance_dict = feature_importance.to_dict()
                mlflow.log_params({
                    f'feature_importance_{k}': v 
                    for k, v in importance_dict.items()
                })
            
            self.logger.info("Training completed successfully")
            self.logger.info(f"Test metrics: {test_metrics}")
            
            return model 