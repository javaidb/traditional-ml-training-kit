import mlflow
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        
        # Set up MLflow experiment
        mlflow.set_experiment(self.experiment_name)
    
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
    
    def _log_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """Log feature importance to MLflow."""
        try:
            # Get feature importance
            importance = model.get_feature_importance()
            if importance is None:
                self.logger.warning("Model does not support feature importance")
                return
            
            # Log importance values
            for feature, imp in importance.items():
                mlflow.log_metric(f'feature_importance_{feature}', imp)
                
        except Exception as e:
            self.logger.warning(f"Could not log feature importance: {str(e)}")
    
    def _log_prediction_plots(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> None:
        """Log prediction metrics to MLflow."""
        try:
            # Log prediction metrics
            residuals = y_true - y_pred
            mlflow.log_metrics({
                f'{prefix}_mean_residual': np.mean(residuals),
                f'{prefix}_std_residual': np.std(residuals),
                f'{prefix}_min_residual': np.min(residuals),
                f'{prefix}_max_residual': np.max(residuals)
            })
            
        except Exception as e:
            self.logger.warning(f"Could not log prediction metrics: {str(e)}")
    
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
        
        # Log trial metrics
        mlflow.log_metrics(metrics, step=trial.number)
        
        # Return the metric to optimize
        return metrics['rmse']  # Assuming we want to minimize RMSE
    
    def _log_plotly_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """Log feature importance plot using Plotly."""
        try:
            importance = model.get_feature_importance()
            if importance is None:
                self.logger.warning("Model does not support feature importance")
                return
            
            # Convert to dictionary if it's a pandas Series
            if isinstance(importance, pd.Series):
                importance_dict = importance.to_dict()
            else:
                importance_dict = importance
            
            # Create feature importance plot
            fig = px.bar(
                x=list(importance_dict.keys()),
                y=list(importance_dict.values()),
                title='Feature Importance',
                labels={'x': 'Feature', 'y': 'Importance'},
                template='plotly_white'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=600,
                showlegend=False
            )
            
            # Log the plot
            mlflow.log_figure(fig, "feature_importance.html")
            
            # Also log the importance values as metrics
            for feature, imp in importance_dict.items():
                mlflow.log_metric(f'feature_importance_{feature}', float(imp))
            
        except Exception as e:
            self.logger.warning(f"Could not log feature importance plot: {str(e)}")

    def _log_plotly_prediction_plots(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> None:
        """Log prediction plots using Plotly."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f'{prefix.capitalize()} Set: Actual vs Predicted',
                    f'{prefix.capitalize()} Set: Residuals Distribution'
                )
            )
            
            # Actual vs Predicted scatter plot
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        size=8,
                        color=y_true,
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=1, col=1
            )
            
            # Add diagonal line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Residuals histogram
            residuals = y_true - y_pred
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name='Residuals',
                    nbinsx=30,
                    marker_color='rgb(55, 83, 109)'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Log the plot
            mlflow.log_figure(fig, f"{prefix}_predictions.html")
            
        except Exception as e:
            self.logger.warning(f"Could not log prediction plots: {str(e)}")

    def _log_plotly_optimization_history(self, study: optuna.Study) -> None:
        """Log optimization history plots using Plotly."""
        try:
            # Create optimization history plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Optimization History',
                    'Parameter Importance'
                )
            )
            
            # Plot optimization history
            trials_df = study.trials_dataframe()
            fig.add_trace(
                go.Scatter(
                    x=trials_df.index,
                    y=trials_df['value'],
                    mode='lines+markers',
                    name='RMSE',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add best value line
            best_value = study.best_value
            fig.add_hline(
                y=best_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Best: {best_value:.2f}",
                row=1, col=1
            )
            
            # Plot parameter importance
            param_importance = optuna.importance.get_param_importances(study)
            fig.add_trace(
                go.Bar(
                    x=list(param_importance.keys()),
                    y=list(param_importance.values()),
                    name='Parameter Importance'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Log the plot
            mlflow.log_figure(fig, "optimization_history.html")
            
        except Exception as e:
            self.logger.warning(f"Could not log optimization history plot: {str(e)}")

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
                'test_ratio': data_config.get('test_ratio', 0.15),
                'n_features': len(self.data_handler.feature_names),
                'n_samples': len(self.data_handler.X_train) + len(self.data_handler.X_val) + len(self.data_handler.X_test)
            })
            
            # Log feature names
            mlflow.log_param('feature_names', ','.join(self.data_handler.feature_names))
            
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
                
                # Log optimization history plots
                self._log_plotly_optimization_history(study)
                
                # Log optimization history CSV
                optimization_history = pd.DataFrame(study.trials_dataframe())
                optimization_history.to_csv('optimization_history.csv')
                mlflow.log_artifact('optimization_history.csv')
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
            
            # Log prediction plots
            self._log_plotly_prediction_plots(
                self.data_handler.y_train,
                model.predict(self.data_handler.X_train),
                'train'
            )
            self._log_plotly_prediction_plots(
                self.data_handler.y_val,
                model.predict(self.data_handler.X_val),
                'val'
            )
            self._log_plotly_prediction_plots(
                self.data_handler.y_test,
                model.predict(self.data_handler.X_test),
                'test'
            )
            
            # Log feature importance
            self._log_plotly_feature_importance(model, self.data_handler.feature_names)
            
            # Log model with signature
            try:
                # Create a sample input for the model signature
                sample_input = self.data_handler.X_train.iloc[:5]
                
                # Log the model with signature
                mlflow.sklearn.log_model(
                    model.model, 
                    "model",
                    input_example=sample_input,
                    registered_model_name=f"{self.config['model']['type']}_{self.experiment_name}"
                )
            except Exception as e:
                self.logger.warning(f"Could not log model with signature: {str(e)}")
                # Fallback to basic model logging
                mlflow.sklearn.log_model(model.model, "model")
            
            self.logger.info("Training completed successfully")
            self.logger.info(f"Test metrics: {test_metrics}")
            
            return model 