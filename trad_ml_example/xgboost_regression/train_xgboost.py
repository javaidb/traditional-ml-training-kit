import os
from trad_ml_training_kit.core.utils import load_config, update_config
from trad_ml_training_kit.core.training import ModelTrainer
import mlflow
import yaml
from datetime import datetime
import pytz
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load base configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    # Load and check data
    data_path = config['data_module']['csv_path']
    df = pd.read_csv(data_path)
    logger.info(f"Original columns: {df.columns.tolist()}")
    logger.info(f"Original data types:\n{df.dtypes}")
    logger.info(f"Sample of neighborhood values:\n{df['neighborhood'].value_counts().head()}")
    
    # Modify configuration as needed
    config_updates = {
        'trainer': {
            'hyperparameter_tuning': {
                'n_trials': 30  # Increase number of trials
            }
        }
    }
    config = update_config(config, config_updates)
    
    # Set MLflow tracking URI using environment variable or default to localhost
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"Using MLflow tracking URI: {mlflow_uri}")
    
    # Set the experiment name from config
    experiment_name = config['experiment']['name']
    mlflow.set_experiment(experiment_name)
    
    # Train model
    trainer = ModelTrainer(config=config)
    model = trainer.train()
    return model

if __name__ == "__main__":
    main() 