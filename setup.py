from setuptools import setup, find_packages

setup(
    name         = "weather-forecast-mlops",
    version      = "2.0.0",
    author       = "Sharif Abusad",
    description  = "MLOps weather forecasting — BiLSTM + XGBoost + FastAPI",
    packages     = find_packages(),
    python_requires = ">=3.10",
)