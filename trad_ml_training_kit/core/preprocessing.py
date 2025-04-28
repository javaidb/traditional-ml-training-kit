"""Data preprocessing and quality checks."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing and quality checks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary containing preprocessing settings
        """
        self.config = config
        self.data_config = config['data_module']
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
            
        Raises:
            ValueError: If missing ratio exceeds configured threshold
        """
        missing_config = self.data_config['data_quality']['missing_values']
        max_missing = missing_config['max_missing_ratio']
        strategy = missing_config['strategy']
        
        # Check missing value ratios
        missing_ratios = df.isnull().mean()
        if (missing_ratios > max_missing).any():
            problematic_cols = missing_ratios[missing_ratios > max_missing].index.tolist()
            raise ValueError(
                f"Columns {problematic_cols} exceed maximum missing ratio of {max_missing}"
            )
        
        # Handle missing values by column type
        for col in df.columns:
            if df[col].isnull().any():
                if col not in self.imputers:
                    if df[col].dtype in ['int64', 'float64']:
                        self.imputers[col] = SimpleImputer(strategy=strategy)
                    else:
                        self.imputers[col] = SimpleImputer(strategy='most_frequent')
                        
                df[col] = self.imputers[col].fit_transform(df[col].values.reshape(-1, 1))
                
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numerical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled outliers
        """
        outlier_config = self.data_config['data_quality']['outlier_detection']
        if not outlier_config['enabled']:
            return df
            
        method = outlier_config['method']
        threshold = outlier_config['threshold']
        
        # Initialize outlier masks dictionary if not exists
        if not hasattr(self, 'outlier_masks'):
            self.outlier_masks = {}
        
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:  # zscore
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
                
            if outliers.any():
                logger.warning(
                    f"Found {outliers.sum()} outliers in column {col} "
                    f"using {method} method"
                )
                # Store outlier mask
                self.outlier_masks[col] = outliers
                
        return df
    
    def _scale_features(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: Input DataFrame
            numerical_columns: List of numerical column names
            fit: Whether to fit the scalers or just transform
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        scaling_config = self.data_config['preprocessing']['scaling']
        method = scaling_config['method']
        
        if not numerical_columns:
            return df
            
        if fit:
            if method == 'standard':
                self.scalers['numerical'] = StandardScaler()
            elif method == 'minmax':
                self.scalers['numerical'] = MinMaxScaler()
            elif method == 'robust':
                self.scalers['numerical'] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
                
        if 'numerical' in self.scalers:
            df_copy = df.copy()
            if fit:
                df_copy[numerical_columns] = self.scalers['numerical'].fit_transform(
                    df_copy[numerical_columns]
                )
            else:
                df_copy[numerical_columns] = self.scalers['numerical'].transform(
                    df_copy[numerical_columns]
                )
            return df_copy
            
        return df
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            fit: Whether to fit the encoders or just transform
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        encoding_config = self.data_config['preprocessing']['encoding']
        method = encoding_config['method']
        max_categories = encoding_config['max_categories']
        
        if not categorical_columns:
            return df
            
        df_encoded = df.copy()
        logger.info(f"Starting categorical encoding for columns: {categorical_columns}")
        logger.info(f"Input columns: {df.columns.tolist()}")
        
        for col in categorical_columns:
            logger.info(f"Processing column: {col}")
            if col not in df.columns:
                logger.error(f"Column {col} not found in DataFrame")
                logger.error(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Column {col} not found in DataFrame")
                
            if fit:
                # Group rare categories
                value_counts = df[col].value_counts()
                logger.info(f"Value counts for {col}: {value_counts}")
                if len(value_counts) > max_categories:
                    top_categories = value_counts.nlargest(max_categories - 1).index
                    df_encoded[col] = df[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                
            if method == 'one_hot':
                if fit:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    self.encoders[col] = {
                        category: f"{col}_{category}"
                        for category in df_encoded[col].unique()
                    }
                    logger.info(f"Created one-hot encoded columns for {col}: {dummies.columns.tolist()}")
                else:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    missing_cols = set(self.encoders[col].values()) - set(dummies.columns)
                    for missing_col in missing_cols:
                        dummies[missing_col] = 0
                    dummies = dummies[list(self.encoders[col].values())]
                    
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                logger.info(f"After encoding {col}, columns are: {df_encoded.columns.tolist()}")
                
            elif method == 'label':
                if fit:
                    unique_values = df_encoded[col].unique()
                    self.encoders[col] = {
                        val: idx for idx, val in enumerate(unique_values)
                    }
                df_encoded[col] = df_encoded[col].map(self.encoders[col])
                
        logger.info(f"Final encoded columns: {df_encoded.columns.tolist()}")
        return df_encoded
    
    def _select_features(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on configured method.
        
        Args:
            df: Input DataFrame
            numerical_columns: List of numerical column names
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: DataFrame with selected features and list of selected columns
        """
        feature_selection = self.data_config['preprocessing']['feature_selection']
        if not feature_selection['enabled']:
            return df, df.columns.tolist()
            
        method = feature_selection['method']
        threshold = feature_selection['threshold']
        
        if method == 'correlation':
            corr_matrix = df[numerical_columns].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [
                column for column in upper.columns
                if any(upper[column] > threshold)
            ]
            logger.info(f"Dropping {len(to_drop)} features due to high correlation")
            return df.drop(to_drop, axis=1), df.drop(to_drop, axis=1).columns.tolist()
            
        elif method == 'variance':
            selector = VarianceThreshold(threshold=threshold)
            selected = selector.fit_transform(df[numerical_columns])
            selected_cols = df[numerical_columns].columns[selector.get_support()].tolist()
            non_numerical = [col for col in df.columns if col not in numerical_columns]
            selected_cols.extend(non_numerical)
            return df[selected_cols], selected_cols
            
        return df, df.columns.tolist()
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        numerical_columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Fit and transform the data.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Processed DataFrame and list of feature names
        """
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._detect_outliers(df)
        
        # Scale numerical features
        df = self._scale_features(df, numerical_columns, fit=True)
        
        # Encode categorical features
        df = self._encode_categorical(df, categorical_columns, fit=True)
        
        # Verify no object columns remain
        object_columns = df.select_dtypes(include=['object']).columns
        if not object_columns.empty:
            logger.warning(f"Found remaining object columns after encoding: {object_columns}")
            # Convert any remaining object columns to category
            for col in object_columns:
                df[col] = df[col].astype('category')
        
        # Select features
        df, selected_features = self._select_features(df, numerical_columns)
        
        return df, selected_features
    
    def transform(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        numerical_columns: List[str]
    ) -> pd.DataFrame:
        """Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        df = self._detect_outliers(df)
        
        # Scale numerical features
        df = self._scale_features(df, numerical_columns, fit=False)
        
        # Encode categorical features
        df = self._encode_categorical(df, categorical_columns, fit=False)
        
        # Verify no object columns remain
        object_columns = df.select_dtypes(include=['object']).columns
        if not object_columns.empty:
            logger.warning(f"Found remaining object columns after encoding: {object_columns}")
            # Convert any remaining object columns to category
            for col in object_columns:
                df[col] = df[col].astype('category')
        
        return df 