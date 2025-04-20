FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training kit and example code
COPY trad_ml_training_kit /app/trad_ml_training_kit
COPY trad_ml_example /app/trad_ml_example

# Set working directory to the example project
WORKDIR /app/trad_ml_example/xgboost_regression

# Generate data and run training
CMD ["bash", "-c", "python generate_data.py && python train_xgboost.py"] 