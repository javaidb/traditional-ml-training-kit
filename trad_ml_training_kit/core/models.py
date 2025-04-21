from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor

class BaseModel(ABC):
    """Base class for all traditional ML models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        pass

class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize XGBoost model.
        
        Args:
            params: XGBoost parameters
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is not installed. Install it with 'pip install xgboost'")
        
        self.params = params
        self.model = xgb.XGBRegressor(**params)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost model."""
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)

class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize LightGBM model.
        
        Args:
            params: LightGBM parameters
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is not installed. Install it with 'pip install lightgbm'")
        
        self.params = params
        self.model = lgb.LGBMRegressor(**params)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LightGBM model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)

class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize CatBoost model.
        
        Args:
            params: CatBoost parameters
        """
        try:
            import catboost as cb
        except ImportError:
            raise ImportError("CatBoost is not installed. Install it with 'pip install catboost'")
        
        self.params = params
        self.model = cb.CatBoostRegressor(**params)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit CatBoost model."""
        self.feature_names = X.columns
        self.model.fit(X, y, verbose=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)

class RandomForestModel(BaseModel):
    """Random Forest model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Random Forest model.
        
        Args:
            params: Random Forest parameters
        """
        self.params = params
        self.model = RandomForestRegressor(**params)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Random Forest model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names)

class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Linear Regression model.
        
        Args:
            params: Linear Regression parameters
        """
        self.params = params
        self.model = LinearRegression(**params)
        self.feature_names = None
        self.coefficients = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Linear Regression model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
        self.coefficients = self.model.coef_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance (coefficients for linear models)."""
        if self.coefficients is None:
            return None
        return pd.Series(np.abs(self.coefficients), index=self.feature_names)

class LassoModel(BaseModel):
    """Lasso Regression model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Lasso model.
        
        Args:
            params: Lasso parameters
        """
        self.params = params
        self.model = Lasso(**params)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Lasso model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance (coefficients for linear models)."""
        return pd.Series(np.abs(self.model.coef_), index=self.feature_names)

class KNNModel(BaseModel):
    """K-Nearest Neighbors model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize KNN model.
        
        Args:
            params: KNN parameters
        """
        self.params = params
        self.model = KNeighborsRegressor(**params)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit KNN model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """KNN does not have built-in feature importance."""
        return None

def get_model_by_name(model_type: str, params: Dict[str, Any]) -> BaseModel:
    """
    Get model instance by type name.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost', 'randomforest', 
                                 'linear', 'lasso', 'knn')
        params: Model parameters
        
    Returns:
        Model instance
    """
    models = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'randomforest': RandomForestModel,
        'linear': LinearRegressionModel,
        'lasso': LassoModel,
        'knn': KNNModel
    }
    
    model_type_lower = model_type.lower()
    if model_type_lower not in models:
        available_models = ", ".join(models.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
    
    return models[model_type_lower](params) 