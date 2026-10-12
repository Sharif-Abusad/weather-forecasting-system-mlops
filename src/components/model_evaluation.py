"""
Model Evaluation Module

Evaluates both trained models against acceptance thresholds:
1. Loads LSTM and XGBoost models from artifact directory.
2. Evaluates LSTM — MAE, RMSE, R² on held-out test set.
3. Evaluates XGBoost — AUC, F1, Accuracy on held-out test set.
4. Compares metrics against thresholds from config.
5. Generates evaluation plots (ROC curve, forecast plot).
6. Saves evaluation report as JSON.
7. Logs results to MLflow.
8. Returns ModelEvaluationArtifact with acceptance decision.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import tensorflow as tf

from xgboost import XGBClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, accuracy_score,
    roc_curve
)
from sklearn.preprocessing import MinMaxScaler

from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from src.utils.main_utils import read_yaml_file, save_json, create_directories


class ModelEvaluation:

    def __init__(
        self,
        config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.config                 = config
        self.model_trainer_artifact = model_trainer_artifact
        self.params                 = read_yaml_file("params.yaml")

    # ── Load models ──────────────────────────────────────────
    def _load_models(self):
        try:
            xgb_model = XGBClassifier()
            xgb_model.load_model(self.model_trainer_artifact.xgb_model_path)

            lstm_model = tf.keras.models.load_model(
                self.model_trainer_artifact.lstm_model_path,
                safe_mode=False,
            )
            scaler = joblib.load(self.model_trainer_artifact.scaler_path)

            with open(self.model_trainer_artifact.lstm_meta_path) as f:
                lstm_meta = json.load(f)
            with open(self.model_trainer_artifact.xgb_meta_path) as f:
                xgb_meta = json.load(f)

            logger.info("Models loaded successfully")
            return xgb_model, lstm_model, scaler, lstm_meta, xgb_meta

        except Exception as e:
            raise WeatherException(e, sys)

    # ── Evaluate XGBoost ─────────────────────────────────────
    def _evaluate_xgboost(
        self,
        model: XGBClassifier,
        meta: dict,
        df: pd.DataFrame,
    ) -> dict:
        try:
            features  = meta["features"]
            threshold = meta["best_threshold"]
            target    = "rain_tomorrow"

            split_idx = int(len(df) * (1 - self.params["split"]["test_size"]))
            X_test    = df[features].iloc[split_idx:]
            y_test    = df[target].astype(int).iloc[split_idx:]

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, zero_division=0)
            auc_score = roc_auc_score(y_test, y_prob)

            # ROC curve plot
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, color="#3b9eff", lw=2, label=f"AUC = {auc_score:.4f}")
            ax.plot([0,1],[0,1],"--", color="gray")
            ax.set_title("XGBoost ROC Curve — Evaluation", fontweight="bold")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend()
            plt.tight_layout()
            roc_path = os.path.join(self.config.artifact_dir, "xgb_roc_curve.png")
            plt.savefig(roc_path, bbox_inches="tight")
            plt.close()

            logger.info(
                f"XGBoost Eval — AUC: {auc_score:.4f} | "
                f"F1: {f1:.4f} | ACC: {acc:.4f} | "
                f"Threshold: {threshold}"
            )

            return {
                "roc_auc"  : round(float(auc_score), 4),
                "f1_score" : round(float(f1),        4),
                "accuracy" : round(float(acc),       4),
                "threshold": threshold,
                "test_rows": int(len(X_test)),
                "roc_plot" : roc_path,
            }

        except Exception as e:
            raise WeatherException(e, sys)

    # ── Evaluate LSTM ────────────────────────────────────────
    def _evaluate_lstm(
        self,
        model,
        scaler: MinMaxScaler,
        meta: dict,
        df: pd.DataFrame,
    ) -> dict:
        try:
            features   = meta["features"]
            lookback   = meta["lookback"]
            forecast_n = meta["forecast_n"]
            n_features = meta["n_features"]
            target_idx = meta["target_idx"]

            df_model = (
                df[["valid_time"] + features]
                .dropna()
                .sort_values("valid_time")
                .reset_index(drop=True)
            )

            scaled = scaler.transform(df_model[features].values).astype(np.float32)

            # Rebuild test sequences
            X_all, y_all = [], []
            for i in range(len(scaled) - lookback - forecast_n + 1):
                X_all.append(scaled[i: i + lookback])
                y_all.append(scaled[i + lookback: i + lookback + forecast_n, target_idx])
            X_all = np.array(X_all, dtype=np.float32)
            y_all = np.array(y_all, dtype=np.float32)

            n       = len(X_all)
            val_end = int(n * 0.85)
            X_test  = X_all[val_end:]
            y_test  = y_all[val_end:]

            y_pred_scaled = model.predict(X_test, verbose=0)

            def inverse_temp(seq):
                dummy = np.zeros((len(seq), n_features))
                dummy[:, target_idx] = seq
                return scaler.inverse_transform(dummy)[:, target_idx]

            y_true_C = np.array([inverse_temp(y_test[i])        for i in range(len(y_test))])
            y_pred_C = np.array([inverse_temp(y_pred_scaled[i]) for i in range(len(y_pred_scaled))])

            mae  = mean_absolute_error(y_true_C.flatten(), y_pred_C.flatten())
            rmse = np.sqrt(mean_squared_error(y_true_C.flatten(), y_pred_C.flatten()))
            r2   = r2_score(y_true_C.flatten(), y_pred_C.flatten())

            # Forecast plot
            n_show = min(200, len(y_true_C))
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(range(n_show), y_true_C[:n_show, 0], label="Actual",    color="#E85D5D", lw=1.5)
            ax.plot(range(n_show), y_pred_C[:n_show, 0], label="Pred +1h",  color="#4A90D9", lw=1.5, ls="--")
            ax.set_title("LSTM Temperature Forecast — Evaluation", fontweight="bold")
            ax.set_ylabel("Temperature (°C)"); ax.legend()
            plt.tight_layout()
            forecast_plot = os.path.join(self.config.artifact_dir, "lstm_forecast_eval.png")
            plt.savefig(forecast_plot, bbox_inches="tight")
            plt.close()

            logger.info(
                f"LSTM Eval — MAE: {mae:.4f}°C | "
                f"RMSE: {rmse:.4f}°C | R²: {r2:.4f}"
            )

            return {
                "mae_celsius"  : round(float(mae),  4),
                "rmse_celsius" : round(float(rmse), 4),
                "r2_score"     : round(float(r2),   4),
                "test_samples" : int(len(X_test)),
                "forecast_plot": forecast_plot,
            }

        except Exception as e:
            raise WeatherException(e, sys)

    # ── Orchestrator ─────────────────────────────────────────
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Evaluation")
            logger.info("=" * 60)

            # Load features
            path = (
                self.model_trainer_artifact
                .lstm_model_path
                .replace(
                    os.path.basename(self.model_trainer_artifact.lstm_model_path), ""
                )
                .replace("model_trainer", "data_transformation")
                + "weather_final.csv"
            )
            df = pd.read_csv(path, parse_dates=["valid_time"])
            df = df.sort_values("valid_time").reset_index(drop=True)

            create_directories([self.config.artifact_dir])

            xgb_model, lstm_model, scaler, lstm_meta, xgb_meta = self._load_models()

            xgb_results  = self._evaluate_xgboost(xgb_model, xgb_meta, df)
            lstm_results = self._evaluate_lstm(lstm_model, scaler, lstm_meta, df)

            # Acceptance check
            xgb_accepted  = xgb_results["roc_auc"] >= self.config.xgb_auc_threshold
            lstm_accepted = lstm_results["mae_celsius"] <= self.config.lstm_mae_threshold
            is_accepted   = xgb_accepted and lstm_accepted

            logger.info(
                f"XGBoost accepted : {xgb_accepted} "
                f"(AUC {xgb_results['roc_auc']} >= {self.config.xgb_auc_threshold})"
            )
            logger.info(
                f"LSTM accepted    : {lstm_accepted} "
                f"(MAE {lstm_results['mae_celsius']} <= {self.config.lstm_mae_threshold})"
            )

            report = {
                "is_model_accepted" : is_accepted,
                "xgboost"           : xgb_results,
                "lstm"              : lstm_results,
                "thresholds": {
                    "xgb_auc_min"  : self.config.xgb_auc_threshold,
                    "lstm_mae_max" : self.config.lstm_mae_threshold,
                },
            }
            save_json(self.config.report_file_path, report)

            # MLflow
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment("weather_model_evaluation")
            with mlflow.start_run(run_name="model_evaluation"):
                mlflow.set_tag("stage", "evaluation")
                mlflow.log_metrics({
                    "eval_xgb_roc_auc"    : xgb_results["roc_auc"],
                    "eval_xgb_f1"         : xgb_results["f1_score"],
                    "eval_lstm_mae"       : lstm_results["mae_celsius"],
                    "eval_lstm_rmse"      : lstm_results["rmse_celsius"],
                    "eval_lstm_r2"        : lstm_results["r2_score"],
                    "model_accepted"      : int(is_accepted),
                })
                mlflow.log_artifact(self.config.report_file_path)
                try:
                    mlflow.log_artifact(xgb_results["roc_plot"])
                    mlflow.log_artifact(lstm_results["forecast_plot"])
                except Exception:
                    pass

            artifact = ModelEvaluationArtifact(
                is_model_accepted = is_accepted,
                lstm_mae          = lstm_results["mae_celsius"],
                xgb_auc           = xgb_results["roc_auc"],
                report_file_path  = self.config.report_file_path,
                message=(
                    "Models accepted." if is_accepted
                    else f"Models rejected — "
                         f"XGB AUC: {xgb_results['roc_auc']} | "
                         f"LSTM MAE: {lstm_results['mae_celsius']}"
                ),
            )

            logger.info("=" * 60)
            logger.info(f"Model Evaluation Artifact: {artifact}")
            logger.info("=" * 60)
            return artifact

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig, DataTransformationConfig,
        DataValidationConfig, ModelTrainerConfig,
        ModelEvaluationConfig,
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    from src.components.data_validation import DataValidation
    from src.components.model_trainer import ModelTrainer

    ingestion_artifact      = DataIngestion(DataIngestionConfig()).initiate_data_ingestion()
    transformation_artifact = DataTransformation(DataTransformationConfig(), ingestion_artifact).initiate_data_transformation()
    validation_artifact     = DataValidation(DataValidationConfig(), transformation_artifact).initiate_data_validation()
    trainer_artifact        = ModelTrainer(ModelTrainerConfig(), validation_artifact).initiate_model_trainer()

    evaluator = ModelEvaluation(ModelEvaluationConfig(), trainer_artifact)
    artifact  = evaluator.initiate_model_evaluation()