FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml /app/

# Install dependencies first - this layer will be cached unless pyproject.toml changes
RUN pip install --no-cache-dir -e .

# Copy source code after installing dependencies
# These layers will only rebuild when the source code changes
COPY trad_ml_training_kit /app/trad_ml_training_kit/
COPY trad_ml_example /app/trad_ml_example/

# Set working directory to the example project and prepare data directory
WORKDIR /app/trad_ml_example/xgboost_regression
RUN mkdir -p temp_data && chmod 777 temp_data

# Generate data and run training
CMD ["bash", "-c", "python generate_data.py && python train_xgboost.py"]