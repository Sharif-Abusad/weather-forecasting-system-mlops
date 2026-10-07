import os
import sys
import yaml
import json
import joblib
import numpy as np
from pathlib import Path
from src.logger import logger
from src.exception import WeatherException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise WeatherException(e, sys)


def write_yaml_file(file_path: str, content: dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise WeatherException(e, sys)


def save_json(file_path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved → {file_path}")
    except Exception as e:
        raise WeatherException(e, sys)


def load_json(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise WeatherException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved → {file_path}")
    except Exception as e:
        raise WeatherException(e, sys)


def load_object(file_path: str) -> object:
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise WeatherException(e, sys)


def create_directories(paths: list) -> None:
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory created → {path}")
    except Exception as e:
        raise WeatherException(e, sys)