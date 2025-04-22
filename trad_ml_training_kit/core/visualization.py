"""Data visualization utilities."""

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Handles creation of data visualization plots."""
    
    def __init__(self, output_dir: str = "data_distributions"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        logger.info(f"Initializing DataVisualizer with output directory: {self.output_dir}")
        self._ensure_dirs()
        
    def _ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        try:
            (self.output_dir / "numerical").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "categorical").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "quality").mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory structure in {self.output_dir}")
            logger.debug(f"Directory contents: {list(self.output_dir.glob('**/*'))}")
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            raise

    def plot_numerical_distribution(
        self,
        data: pd.DataFrame,
        column: str,
        outlier_mask: Optional[pd.Series] = None,
        original_data: Optional[pd.Series] = None
    ) -> None:
        """Plot distribution of numerical features with outliers highlighted.
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot
            outlier_mask: Boolean mask indicating outliers
            original_data: Original data before preprocessing
        """
        plt.figure(figsize=(12, 6))
        
        # Plot original data if available
        if original_data is not None:
            plt.subplot(1, 2, 1)
            sns.histplot(original_data, kde=True, label='Original')
            plt.title(f'Original Distribution - {column}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            
            # Plot outliers if mask is available
            if outlier_mask is not None:
                # Ensure the mask and data have the same index and handle NaN values
                outlier_mask = outlier_mask.reindex(original_data.index).fillna(False).astype(bool)
                outliers = original_data[outlier_mask]
                if len(outliers) > 0:
                    plt.scatter(outliers, [0] * len(outliers), color='red', 
                              label='Outliers', alpha=0.5)
                    plt.legend()
            
            # Plot preprocessed data
            plt.subplot(1, 2, 2)
            
        sns.histplot(data[column], kde=True, label='Preprocessed')
        plt.title(f'{"Preprocessed " if original_data is not None else ""}Distribution - {column}')
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        plt.tight_layout()
        self._save_plot(f'distribution_{column}')

    def plot_categorical_distribution(
        self,
        data: pd.DataFrame,
        column: str,
        encoding_map: Optional[Dict] = None,
        original_data: Optional[pd.Series] = None
    ) -> None:
        """Plot distribution of categorical features with encoding information.
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot
            encoding_map: Dictionary mapping categories to encoded values
            original_data: Original data before preprocessing
        """
        plt.figure(figsize=(12, 6))
        
        # Plot original distribution if available
        if original_data is not None:
            plt.subplot(1, 2, 1)
            value_counts = original_data.value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Original Distribution - {column}')
            plt.xticks(rotation=45)
            
            # Add encoding information if available
            if encoding_map:
                plt.subplot(1, 2, 2)
                
        # Plot encoded distribution
        if encoding_map:
            # For one-hot encoded columns
            if isinstance(encoding_map, dict) and all(isinstance(v, str) for v in encoding_map.values()):
                encoded_cols = [col for col in data.columns if col.startswith(f'{column}_')]
                values = data[encoded_cols].sum()
                plt.bar(range(len(values)), values)
                plt.xticks(range(len(values)), 
                          [col.replace(f'{column}_', '') for col in encoded_cols],
                          rotation=45)
            # For label encoded columns
            else:
                value_counts = data[column].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                # Add encoding mapping as text
                for i, (cat, code) in enumerate(encoding_map.items()):
                    plt.text(i, 0, f'{cat}: {code}', rotation=45)
                    
        plt.title(f'{"Encoded " if original_data is not None else ""}Distribution - {column}')
        plt.tight_layout()
        self._save_plot(f'distribution_{column}')

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        original_data: Optional[pd.DataFrame] = None
    ) -> None:
        """Plot correlation matrix before and after preprocessing.
        
        Args:
            data: DataFrame containing the preprocessed data
            original_data: DataFrame containing the original data
        """
        if original_data is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Original correlation matrix
            corr_orig = original_data.corr()
            sns.heatmap(corr_orig, annot=True, cmap='coolwarm', ax=ax1)
            ax1.set_title('Original Correlation Matrix')
            
            # Preprocessed correlation matrix
            corr_proc = data.corr()
            sns.heatmap(corr_proc, annot=True, cmap='coolwarm', ax=ax2)
            ax2.set_title('Preprocessed Correlation Matrix')
        else:
            plt.figure(figsize=(10, 8))
            corr = data.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            
        plt.tight_layout()
        self._save_plot('correlation_matrix')

    def plot_feature_importance(
        self,
        importance_scores: np.ndarray,
        feature_names: List[str],
        top_n: int = 20
    ) -> None:
        """Plot feature importance scores with actual feature names.
        
        Args:
            importance_scores: Array of feature importance scores
            feature_names: List of feature names
            top_n: Number of top features to display
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]
        
        # Select top N features
        if len(indices) > top_n:
            indices = indices[:top_n]
            
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(indices)), importance_scores[indices])
        plt.xticks(range(len(indices)), 
                  [feature_names[i] for i in indices],
                  rotation=45, ha='right')
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        self._save_plot('feature_importance')

    def plot_target_distribution(
        self,
        data: pd.Series,
        name: str
    ) -> None:
        """Create distribution plot for target variable.
        
        Args:
            data: Target variable data
            name: Target variable name
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, kde=True)
        plt.title(f"Distribution of Target Variable ({name})")
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.tight_layout()
        self._save_plot("target_distribution")

    def _save_plot(self, name: str) -> None:
        """Save the current plot to a file.
        
        Args:
            name: Name of the plot
        """
        try:
            output_path = self.output_dir / f"{name}.png"
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {name}: {str(e)}")
            raise 