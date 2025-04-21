from setuptools import setup, find_packages

setup(
    name="trad_ml_training_kit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "mlflow",
    ],
) 