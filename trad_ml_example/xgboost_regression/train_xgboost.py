import os
from trad_ml_training_kit.core.utils import load_config, update_config
from trad_ml_training_kit.core.training import ModelTrainer
import mlflow
import yaml

def main():
    # Load base configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    
    # Modify configuration as needed
    config_updates = {
        'trainer': {
            'hyperparameter_tuning': {
                'n_trials': 30  # Increase number of trials
            }
        }
    }
    config = update_config(config, config_updates)
    
    # Set MLflow tracking URI using container name
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Train model
    trainer = ModelTrainer(config=config)
    model = trainer.train()
    return model

if __name__ == "__main__":
    main() 