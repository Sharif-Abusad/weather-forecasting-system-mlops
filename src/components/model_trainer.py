"""
Model Trainer Orchestrator

Runs both trainers in sequence under a single parent MLflow run:
1. Loads validated feature dataset.
2. Trains XGBoost rain classifier (nested run).
3. Trains BiLSTM temperature forecaster (nested run).
4. Saves both ModelTrainerArtifacts.
5. Returns a ModelTrainerArtifact for downstream pipeline stages.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import mlflow

from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import (
    ModelTrainerConfig,
    DataValidationConfig,
)
from src.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact
)
from src.utils.main_utils import create_directories
from src.components.xgboost_trainer import XGBoostTrainer
from src.components.lstm_trainer import LSTMTrainer
from src.constants import DATA_TRANSFORMATION_FEATURES_DIR


class ModelTrainer:

    def __init__(
        self,
        config: ModelTrainerConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        self.config                   = config
        self.data_validation_artifact = data_validation_artifact

    def _load_features(self) -> pd.DataFrame:
        try:
            path = os.path.join(DATA_TRANSFORMATION_FEATURES_DIR, "weather_final.csv")
            df = pd.read_csv(path, parse_dates=["valid_time"])
            df = df.sort_values("valid_time").reset_index(drop=True)
            logger.info(f"Features loaded — {len(df):,} rows from {path}")
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Trainer")
            logger.info("=" * 60)

            df = self._load_features()
            create_directories([self.config.artifact_dir])

            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment("weather_full_pipeline")

            with mlflow.start_run(run_name="model_trainer_pipeline") as parent_run:
                mlflow.set_tag("stage", "training")

                # ── XGBoost ──────────────────────────────────
                logger.info("Starting XGBoost training...")
                with mlflow.start_run(run_name="xgboost_rain", nested=True):
                    xgb_result = XGBoostTrainer(self.config).train(df)
                    xgb_meta   = xgb_result["meta"]

                # ── LSTM ──────────────────────────────────────
                logger.info("Starting LSTM training...")
                with mlflow.start_run(run_name="lstm_temperature", nested=True):
                    lstm_result = LSTMTrainer(self.config).train(df)
                    lstm_meta   = lstm_result["meta"]

                # Log summary to parent run
                mlflow.log_metrics({
                    "xgb_roc_auc"  : xgb_meta["roc_auc"],
                    "xgb_f1_tuned" : xgb_meta["metrics_at_best_threshold"]["f1_score"],
                    "lstm_mae"     : lstm_meta["metrics"]["mae_celsius"],
                    "lstm_rmse"    : lstm_meta["metrics"]["rmse_celsius"],
                    "lstm_r2"      : lstm_meta["metrics"]["r2_score"],
                })

            artifact = ModelTrainerArtifact(
                lstm_model_path = self.config.lstm_model_path,
                xgb_model_path  = self.config.xgb_model_path,
                lstm_meta_path  = self.config.lstm_meta_path,
                xgb_meta_path   = self.config.xgb_meta_path,
                scaler_path     = self.config.scaler_path,
                lstm_mae        = lstm_meta["metrics"]["mae_celsius"],
                xgb_auc         = xgb_meta["roc_auc"],
                is_trained      = True,
                message=(
                    f"Training complete. "
                    f"LSTM MAE: {lstm_meta['metrics']['mae_celsius']:.4f}°C | "
                    f"XGBoost AUC: {xgb_meta['roc_auc']:.4f}"
                ),
            )

            logger.info("=" * 60)
            logger.info(f"Model Trainer Artifact: {artifact}")
            logger.info("=" * 60)
            return artifact

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    validation_config = DataValidationConfig()
    report_path = Path(validation_config.report_file_path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Validation report not found: {report_path}"
        )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    validation_status = report["validation_status"]

    validation_artifact = DataValidationArtifact(
        validation_status=validation_status,
        report_file_path=str(report_path),
        message=(
            "Validation passed."
            if validation_status
            else f"Validation failed: {report.get('errors', [])}"
        ),
    )

    if not validation_status:
        raise ValueError(validation_artifact.message)

    trainer = ModelTrainer(
        ModelTrainerConfig(),
        validation_artifact,
    )

    artifact = trainer.initiate_model_trainer()