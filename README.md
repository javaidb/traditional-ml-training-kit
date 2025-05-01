# Traditional ML Training Kit 🤖

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made with Scikit-learn](https://img.shields.io/badge/Made%20with-Scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/Tracked%20with-MLflow-0194E2.svg)](https://mlflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive toolkit for traditional machine learning algorithm implementations, focusing on standardized structure and best practices for regression problems. This project serves as an educational resource and template for implementing various ML algorithms in a consistent, production-ready manner.

## 🎯 Project Overview

This repository contains standardized implementations of common machine learning algorithms, each solving a housing price prediction problem. The project emphasizes:

- Consistent project structure across different algorithms
- Best practices in ML implementation
- Proper configuration management
- Performance tracking with MLflow
- Synthetic data generation for learning purposes

## 🏗️ Project Structure

Each algorithm implementation follows a standardized structure:

```
algorithm_name/
├── generate_data.py      # Synthetic housing data generator
├── train_algorithm.py    # Algorithm-specific training script
├── config.yaml          # Algorithm configuration
├── README.md           # Algorithm-specific documentation
└── temp_data/         # Generated dataset storage
```

## 🧮 Implemented Algorithms

| Algorithm | Status | Best Use Cases |
|-----------|--------|---------------|
| KNN | ✅ Complete | Small to medium datasets, when feature scales are similar |
| Lasso | ✅ Complete | High-dimensional data, feature selection needed |
| Linear Regression | ✅ Complete | Linear relationships, baseline modeling |
| Random Forest | 🚧 Partial | Complex relationships, robust to outliers |

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/javaidb/traditional-ml-training-kit.git
cd traditional-ml-training-kit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Choose an algorithm directory and generate data:
```bash
cd algorithm_name
python generate_data.py
```

4. Train the model:
```bash
python train_algorithm.py
```

## 📊 Data Generation

The synthetic housing dataset includes features such as:
- Square footage
- Number of bedrooms/bathrooms
- Year built
- Lot size
- Neighborhood (categorical)
- Property type (categorical)

## 🔧 Configuration

Each algorithm includes a `config.yaml` file for:
- Data preprocessing parameters
- Model hyperparameters
- Training settings
- MLflow configuration

## 📈 MLflow Integration

The project uses MLflow for:
- Experiment tracking
- Model versioning
- Performance metrics logging
- Hyperparameter optimization results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛠️ Built With

- Python 3.8+
- scikit-learn
- MLflow
- pandas
- numpy
- PyYAML
- Optuna (for hyperparameter tuning)

## 📫 Contact

For questions or feedback, please open an issue in the repository.

---