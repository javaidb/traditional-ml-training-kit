"""Data loading and processing module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import os
import tempfile
import shutil
from datetime import datetime

from .preprocessing import DataPreprocessor
from .visualization import DataVisualizer

logger = logging.getLogger(__name__)

class DataModule:
    """Handles data loading, splitting, and preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data module with configuration.
        
        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.data_config = config['data_module']
        self.preprocessor = DataPreprocessor(config)
        self.feature_names: Optional[List[str]] = None
        
        # Store original data for visualization
        self.original_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
        # Handle relative paths by joining with current working directory
        self.csv_path = os.path.join(os.getcwd(), self.data_config['csv_path']) \
            if not os.path.isabs(self.data_config['csv_path']) \
            else self.data_config['csv_path']
            
        # Validate split ratios if provided
        split_ratios = [
            self.data_config.get('validation', {}).get('train_ratio', 0.7),
            self.data_config.get('validation', {}).get('val_ratio', 0.15),
            self.data_config.get('validation', {}).get('test_ratio', 0.15)
        ]
        if not abs(sum(split_ratios) - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Initialize data splits
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that all specified columns exist in the dataset.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If any specified column is missing
        """
        # Infer numerical columns if not provided
        if not self.data_config['numerical_columns']:
            self.data_config['numerical_columns'] = df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            self.data_config['numerical_columns'] = [
                col for col in self.data_config['numerical_columns']
                if col not in (
                    self.data_config['categorical_columns'] +
                    [self.data_config['target_column']]
                )
            ]
            logger.info(
                f"Inferred numerical columns: {self.data_config['numerical_columns']}"
            )
        
        all_columns = (
            [self.data_config['target_column']] +
            self.data_config['categorical_columns'] +
            self.data_config['numerical_columns']
        )
        
        missing_columns = set(all_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"The following columns are missing from the dataset: {missing_columns}"
            )
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types of columns.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If columns have unexpected data types
        """
        for col in self.data_config['numerical_columns']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Column {col} is specified as numerical but contains non-numeric data"
                )
    
    def load_data(self) -> pd.DataFrame:
        """Load data from specified source.
        
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(
                f"Data file not found at {self.csv_path} "
                f"(working directory: {os.getcwd()})"
            )
            
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded dataset with shape {df.shape}")
        
        self._validate_columns(df)
        self._validate_data_types(df)
        
        return df
    
    def _create_distribution_plots(self) -> str:
        """Create distribution plots for features and target.
        
        Returns:
            str: Path to the directory containing the plots
        """
        if self.original_data is None or self.processed_data is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
            
        # Create temporary directory in /tmp for better Docker compatibility
        temp_dir = "/tmp/data_distributions_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory for plots: {temp_dir}")
        
        visualizer = DataVisualizer(temp_dir)
        
        try:
            # Plot numerical features
            for col in self.data_config['numerical_columns']:
                # Get outlier mask if available
                outlier_mask = None
                if hasattr(self.preprocessor, 'outlier_masks') and col in self.preprocessor.outlier_masks:
                    outlier_mask = self.preprocessor.outlier_masks[col]
                
                # Get missing value mask if available
                missing_mask = self.original_data[col].isnull()
                
                visualizer.plot_numerical_distribution(
                    data=self.processed_data,
                    column=col,
                    outlier_mask=outlier_mask if hasattr(self.preprocessor, 'outlier_masks') and col in self.preprocessor.outlier_masks else None,
                    original_data=self.original_data[col]
                )
            
            # Plot categorical features
            for col in self.data_config['categorical_columns']:
                # Get encoding map if available
                encoding_map = None
                if hasattr(self.preprocessor, 'encoders') and col in self.preprocessor.encoders:
                    encoding_map = self.preprocessor.encoders[col]
                
                visualizer.plot_categorical_distribution(
                    data=self.processed_data,
                    column=col,
                    encoding_map=encoding_map,
                    original_data=self.original_data[col]
                )
            
            # Plot correlation matrices
            numerical_cols = self.data_config['numerical_columns']
            if numerical_cols:
                # For after preprocessing, only include numerical columns
                processed_numerical = [
                    col for col in self.processed_data.columns
                    if col in numerical_cols or
                    not any(col.startswith(f"{cat_col}_") 
                           for cat_col in self.data_config['categorical_columns'])
                ]
                
                # Create correlation matrix
                visualizer.plot_correlation_matrix(
                    data=self.processed_data[processed_numerical],
                    original_data=self.original_data[numerical_cols]
                )
            
            # Plot target distribution
            target_col = self.data_config['target_column']
            visualizer.plot_target_distribution(
                data=self.original_data[target_col],
                name=target_col
            )
            
            logger.info(f"Successfully created all plots in {temp_dir}")
            logger.debug(f"Directory contents: {os.listdir(temp_dir)}")
            
            return temp_dir
        except Exception as e:
            logger.error(f"Failed to create plots: {str(e)}")
            logger.error(f"Original data columns: {self.original_data.columns.tolist()}")
            logger.error(f"Processed data columns: {self.processed_data.columns.tolist()}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise

    def prepare_data(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training.
        
        Args:
            df: Optional input DataFrame. If None, data will be loaded from csv_path.
            
        Returns:
            Tuple containing:
                - X_train: Training features
                - X_val: Validation features
                - X_test: Test features
                - y_train: Training targets
                - y_val: Validation targets
                - y_test: Test targets
                - feature_names: List of feature names
        """
        if df is None:
            df = self.load_data()
            
        # Store original data
        self.original_data = df.copy()
        
        # Split features and target
        X = df.drop(self.data_config['target_column'], axis=1)
        y = df[self.data_config['target_column']]
        
        # Get split ratios
        val_config = self.data_config.get('validation', {})
        test_ratio = val_config.get('test_ratio', 0.15)
        val_ratio = val_config.get('val_ratio', 0.15)
        
        # Get random seed from experiment config
        random_seed = self.config.get('experiment', {}).get('random_seed', 42)
        
        # First split: train + validation vs test
        X_trainval, self.X_test, y_trainval, self.y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_seed
        )
        
        # Second split: train vs validation
        val_ratio_adjusted = val_ratio / (1 - test_ratio)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_adjusted,
            random_state=random_seed
        )
        
        # Preprocess training data
        X_train_processed, feature_names = self.preprocessor.fit_transform(
            self.X_train,
            self.data_config['categorical_columns'],
            self.data_config['numerical_columns']
        )
        
        # Transform validation and test data
        X_val_processed = self.preprocessor.transform(
            self.X_val,
            self.data_config['categorical_columns'],
            self.data_config['numerical_columns']
        )
        
        X_test_processed = self.preprocessor.transform(
            self.X_test,
            self.data_config['categorical_columns'],
            self.data_config['numerical_columns']
        )
        
        # Store processed data
        self.processed_data = pd.concat([
            X_train_processed,
            X_val_processed,
            X_test_processed
        ])
        
        self.feature_names = feature_names
        
        # Create distribution plots
        self.distribution_plots_dir = self._create_distribution_plots()
        
        return (
            X_train_processed.values,
            X_val_processed.values,
            X_test_processed.values,
            self.y_train.values,
            self.y_val.values,
            self.y_test.values,
            feature_names
        )
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        return self.X_train, self.y_train
    
    def get_val_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get validation data."""
        if self.X_val is None or self.y_val is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        return self.X_val, self.y_val
    
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get test data."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
        return self.X_test, self.y_test
    
    def process_inference_data(self, df: pd.DataFrame) -> np.ndarray:
        """Process data for inference.
        
        Args:
            df: Input DataFrame
            
        Returns:
            np.ndarray: Processed features ready for inference
        """
        if self.feature_names is None:
            raise ValueError(
                "Preprocessor has not been fitted. Call prepare_data first."
            )
            
        # Validate input data
        self._validate_columns(df)
        self._validate_data_types(df)
        
        # Extract features
        X = df.drop(self.data_config['target_column'], axis=1)
        
        # Process features
        X_processed = self.preprocessor.transform(
            X,
            self.data_config['categorical_columns'],
            self.data_config['numerical_columns']
        )
        
        return X_processed.values 