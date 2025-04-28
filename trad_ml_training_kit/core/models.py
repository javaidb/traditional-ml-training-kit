from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

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
        """Initialize model with parameters."""
        self.params = params.copy()
        # Enable categorical feature support
        self.params['enable_categorical'] = True
        self.model = xgb.XGBRegressor(**self.params)
        
    def fit(self, X: Any, y: pd.Series) -> None:
        """Fit model to data."""
        # Convert to DataFrame if ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        # Convert object dtypes to category
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')
        self.model.fit(X, y)
        
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions."""
        # Convert to DataFrame if ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        # Convert object dtypes to category
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet")
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))

class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """Initialize model with parameters."""
        self.params = params
        self.model = lgb.LGBMRegressor(**params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model to data."""
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet")
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))

class CatBoostModel(BaseModel):
    """CatBoost model wrapper."""
    
    def __init__(self, params: Dict[str, Any]):
        """Initialize model with parameters."""
        self.params = params
        self.model = cb.CatBoostRegressor(**params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit model to data."""
        self.model.fit(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet")
        return dict(zip(self.model.feature_names_, self.model.feature_importances_))

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

def get_model_by_name(name: str, params: Dict[str, Any]) -> BaseModel:
    """Get model instance by name."""
    if name == 'xgboost':
        return XGBoostModel(params)
    elif name == 'lightgbm':
        return LightGBMModel(params)
    elif name == 'catboost':
        return CatBoostModel(params)
    elif name == 'randomforest':
        return RandomForestModel(params)
    elif name == 'linear':
        return LinearRegressionModel(params)
    elif name == 'lasso':
        return LassoModel(params)
    elif name == 'knn':
        return KNNModel(params)
    else:
        raise ValueError(f"Unknown model type: {name}") 