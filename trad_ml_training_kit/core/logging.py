import mlflow
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
import logging
import tempfile
import os

class MLflowLogger:
    """MLflow experiment tracking wrapper with advanced visualization and logging capabilities."""
    
    def __init__(
        self, 
        experiment_name: str,
        run_name: Optional[str] = None,
        experiment_tags: Dict[str, Any] = {},
        model_artifact_path: str = "model",
        include_date_in_run_name: bool = True
    ):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for the run
            experiment_tags: Optional dictionary of experiment tags
            model_artifact_path: Path where model artifacts will be saved
            include_date_in_run_name: Whether to append timestamp to run name
        """
        self.experiment_name = experiment_name
        self._run_name = run_name
        self._experiment_tags = experiment_tags
        self._model_artifact_path = model_artifact_path
        self._include_date_in_run_name = include_date_in_run_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize timestamp
        self._init_time = datetime.utcnow().strftime("%Y_%m_%d_T%H_%M_%SZ")
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

    @property
    def run_name(self) -> str:
        """Get the formatted run name with optional timestamp."""
        if self._run_name is None:
            return f"run_{self._init_time}"
        elif self._include_date_in_run_name:
            return f"{self._run_name}_{self._init_time}"
        else:
            return self._run_name
            
    @property
    def model_artifact_path(self) -> str:
        """Get the model artifact path."""
        return self._model_artifact_path

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log model parameters."""
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log model metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=step)

    def log_model(
        self,
        model: Any,
        model_name: str,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log the model artifact with optional signature.
        
        Args:
            model: Trained model
            model_name: Name for the model artifact
            input_example: Optional sample input for model signature
            registered_model_name: Optional name for model registration
        """
        try:
            if input_example is not None:
                mlflow.sklearn.log_model(
                    model,
                    model_name,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:
                mlflow.sklearn.log_model(model, model_name)
        except Exception as e:
            self.logger.warning(f"Could not log model with signature: {str(e)}")
            mlflow.sklearn.log_model(model, model_name)

    def log_feature_importance(
        self,
        importance: Union[Dict[str, float], pd.Series],
        feature_names: List[str]
    ) -> None:
        """
        Log feature importance values and plot.
        
        Args:
            importance: Feature importance values as dict or Series
            feature_names: List of feature names
        """
        try:
            # Convert to dictionary if it's a pandas Series and ensure it uses feature_names
            if isinstance(importance, pd.Series):
                importance_dict = {feature_names[i]: val for i, val in enumerate(importance)}
            else:
                importance_dict = {feature_names[i]: val for i, val in enumerate(importance.values())}
            
            # Create feature importance plot
            fig = px.bar(
                x=feature_names,
                y=[importance_dict[f] for f in feature_names],
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
            
            # Log importance values as metrics
            for feature in feature_names:
                # Clean feature name to be MLflow-compatible (replace spaces and special chars with underscores)
                clean_feature = feature.replace(' ', '_').replace('-', '_').replace('/', '_').lower()
                mlflow.log_metric(f'feature_importance_{clean_feature}', float(importance_dict[feature]))
            
        except Exception as e:
            self.logger.warning(f"Could not log feature importance plot: {str(e)}")

    def log_prediction_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str
    ) -> None:
        """
        Log prediction plots and metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            prefix: Prefix for metric names (e.g., 'train', 'val', 'test')
        """
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
            
            # Log prediction metrics
            mlflow.log_metrics({
                f'{prefix}_mean_residual': float(np.mean(residuals)),
                f'{prefix}_std_residual': float(np.std(residuals)),
                f'{prefix}_min_residual': float(np.min(residuals)),
                f'{prefix}_max_residual': float(np.max(residuals))
            })
            
        except Exception as e:
            self.logger.warning(f"Could not log prediction plots: {str(e)}")

    def log_optimization_history(self, study: optuna.Study) -> None:
        """
        Log optimization history plots and metrics.
        
        Args:
            study: Optuna study object containing optimization results
        """
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
                    name='Objective Value',
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
            
            # Log optimization history CSV using a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                optimization_history = pd.DataFrame(study.trials_dataframe())
                optimization_history.to_csv(temp_file.name)
                mlflow.log_artifact(temp_file.name, "optimization_history.csv")
                
            # Clean up temporary file
            os.unlink(temp_file.name)
            
        except Exception as e:
            self.logger.warning(f"Could not log optimization history plot: {str(e)}")

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run."""
        run_name = run_name or self.run_name
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=self._experiment_tags
        )
    
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