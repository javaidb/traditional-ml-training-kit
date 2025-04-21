FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project for development installation
COPY . /app/

# Install Python dependencies and the package in development mode
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Set working directory to the example project
WORKDIR /app/trad_ml_example/xgboost_regression

# Create temp_data directory with proper permissions
RUN mkdir -p temp_data && chmod 777 temp_data

# Debug: List installed packages
RUN pip list

# Generate data and run training
CMD ["bash", "-c", "python generate_data.py && python train_xgboost.py"] 