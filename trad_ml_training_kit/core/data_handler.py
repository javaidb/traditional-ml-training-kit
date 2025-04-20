import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Optional, Tuple, List, Union
import os

class DataHandler:
    """Data handler for traditional ML methods."""
    
    def __init__(
        self,
        csv_path: str,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize the data handler.
        
        Args:
            csv_path: Path to the CSV file
            target_column: Name of the target column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_state: Random state for reproducibility
        """
        self.csv_path = csv_path
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Initialize preprocessing objects
        self.scalers = {}
        self.encoders = {}
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    def setup(self) -> None:
        """Set up the data handler by loading and preprocessing data."""
        # Read data
        self.df = pd.read_csv(self.csv_path)
        
        # Infer numerical columns if not provided
        if self.numerical_columns is None:
            self.numerical_columns = self.df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            self.numerical_columns = [
                col for col in self.numerical_columns 
                if col not in self.categorical_columns + [self.target_column]
            ]
        
        # Preprocess features
        self._preprocess_features()
        
        # Split data
        self._split_data()
    
    def _preprocess_features(self) -> None:
        """Preprocess numerical and categorical features."""
        # Handle numerical features
        for col in self.numerical_columns:
            scaler = StandardScaler()
            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler
        
        # Handle categorical features
        for col in self.categorical_columns:
            encoder = LabelEncoder()
            self.df[col] = encoder.fit_transform(self.df[col])
            self.encoders[col] = encoder
    
    def _split_data(self) -> None:
        """Split data into train, validation, and test sets."""
        # Prepare features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # First split: train + validation vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.random_state
        )
        
        # Second split: train vs validation
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_adjusted,
            random_state=self.random_state
        )
        
        # Store splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Store feature names for model input
        self.feature_names = X_train.columns.tolist()
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data."""
        return self.X_train, self.y_train
    
    def get_val_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get validation data."""
        return self.X_val, self.y_val
    
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get test data."""
        return self.X_test, self.y_test
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors."""
        df = df.copy()
        
        # Apply numerical scaling
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])
        
        # Apply categorical encoding
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        return df 