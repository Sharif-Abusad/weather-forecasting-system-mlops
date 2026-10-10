from dataclasses import dataclass
from src.constants import *

@dataclass
class DataIngestionConfig:
    artifact_dir:  str = os.path.join(ARTIFACT_DIR, TIMESTAMP, DATA_INGESTION_DIR)
    raw_file_path: str = os.path.join(artifact_dir, DATA_INGESTION_RAW_DIR, RAW_FILE_NAME)
    archive_dir:   str = os.path.join("data", "archives")   # ZIPs live here, git ignored
    raw_dir:       str = os.path.join("data", "raw")            # extracted NetCDFs live here


@dataclass
class DataTransformationConfig:
    artifact_dir:        str = os.path.join(ARTIFACT_DIR, TIMESTAMP, DATA_TRANSFORMATION_DIR)
    clean_file_path:     str = os.path.join(artifact_dir, "weather_clean.csv")
    final_features_path: str = os.path.join(artifact_dir, "weather_final.csv")


@dataclass
class DataValidationConfig:
    artifact_dir:     str = os.path.join(ARTIFACT_DIR, TIMESTAMP, DATA_VALIDATION_DIR)
    report_file_path: str = os.path.join(artifact_dir, DATA_VALIDATION_REPORT_FILE)
    schema_file_path: str = SCHEMA_FILE_PATH


@dataclass
class ModelTrainerConfig:
    artifact_dir:     str = os.path.join(ARTIFACT_DIR, TIMESTAMP, MODEL_TRAINER_DIR)
    lstm_model_path:  str = os.path.join(artifact_dir, LSTM_MODEL_FILE)
    xgb_model_path:   str = os.path.join(artifact_dir, XGB_MODEL_FILE)
    lstm_meta_path:   str = os.path.join(artifact_dir, LSTM_META_FILE)
    xgb_meta_path:    str = os.path.join(artifact_dir, XGB_META_FILE)
    scaler_path:      str = os.path.join(artifact_dir, SCALER_FILE_NAME)
    params_file_path: str = PARAMS_FILE_PATH