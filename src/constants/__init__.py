import os
from datetime import datetime

# ── Pipeline artifact root ────────────────────────
ARTIFACT_DIR: str = "artifact"
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# ── Config files ──────────────────────────────────
CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")
PARAMS_FILE_PATH: str = os.path.join("params.yaml")

# ── Data Ingestion ────────────────────────────────
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_INGESTION_RAW_DIR: str = "raw"
DATA_INGESTION_PROCESSED_DIR: str = "processed"
RAW_FILE_NAME: str = "weather_raw.csv"

# Open-Meteo default coords (Azamgarh — override via params.yaml)
DEFAULT_LATITUDE: float = 26.00
DEFAULT_LONGITUDE: float = 83.00
DEFAULT_TIMEZONE: str = "Asia/Kolkata"
DEFAULT_PAST_DAYS: int = 90

# ── Data Validation ───────────────────────────────
DATA_VALIDATION_DIR: str = "data_validation"
DATA_VALIDATION_REPORT_FILE: str = "validation_report.json"

# ── Data Transformation ───────────────────────────
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATA_TRANSFORMATION_FEATURES_DIR: str = "features"
LSTM_FEATURES_FILE: str = "lstm_features.csv"
XGB_FEATURES_FILE: str = "xgb_features.csv"
SCALER_FILE_NAME: str = "lstm_temp_scaler.pkl"

# ── Model Trainer ─────────────────────────────────
MODEL_TRAINER_DIR: str = "model_trainer"
LSTM_MODEL_FILE: str = "lstm_temp_model.keras"
XGB_MODEL_FILE: str = "rain_model.json"
LSTM_META_FILE: str = "lstm_model_meta.json"
XGB_META_FILE: str = "xgb_model_meta.json"

# ── Model Evaluation ──────────────────────────────
MODEL_EVALUATION_DIR: str = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE: str = "eval_report.json"
LSTM_MAE_THRESHOLD: float = 5.0        # reject model if MAE exceeds this
XGB_AUC_THRESHOLD: float = 0.75        # reject model if AUC below this

# ── Model Pusher ──────────────────────────────────
MODEL_PUSHER_DIR: str = "model_pusher"
SAVED_MODEL_DIR: str = "saved_models"

# ── MLflow ────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = "weather_forecast"
LSTM_RUN_NAME: str = "lstm_temperature"
XGB_RUN_NAME: str = "xgboost_rain"

# ── Target columns ────────────────────────────────
TEMPERATURE_TARGET_COL: str = "temperature"
RAIN_TARGET_COL: str = "rain_tomorrow"

# ── LSTM feature list ─────────────────────────────
LSTM_FEATURES: list = [
    "temperature",
    "temp_lag_1",
    "temp_lag_24",
    "temp_rolling_6",
    "humidity",
    "surface_pressure",
    "total_cloud_cover",
    "wind_speed",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

# ── XGBoost feature list ──────────────────────────
XGB_FEATURES: list = [
    "temperature",
    "surface_pressure",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "precipitation",
    "humidity",
    "wind_speed",
    "temp_rolling_6",
    "temp_lag_1",
    "temp_lag_24",
    "month",
]