"""
Model Evaluation Module

Evaluates both trained models with two-stage acceptance logic:
1. If no previous model exists → use fixed thresholds.
2. If previous model exists → accept only if new model is better.

Acceptance criteria:
- LSTM : new MAE < previous MAE  (lower is better)
- XGBoost: new AUC > previous AUC (higher is better)

Fixed thresholds (first run only):
- LSTM MAE  <= 5.0 °C
- XGBoost AUC >= 0.75
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
    roc_auc_score, f1_score, accuracy_score, roc_curve,
)
from sklearn.preprocessing import MinMaxScaler

from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from src.utils.main_utils import read_yaml_file, save_json, create_directories
from src.constants import DATA_TRANSFORMATION_FEATURES_DIR

class ModelEvaluation:

    def __init__(
        self,
        config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.config                 = config
        self.model_trainer_artifact = model_trainer_artifact
        self.params                 = read_yaml_file("params.yaml")

    # ── Load previous metrics from saved_models ──────────────
    def _load_previous_metrics(self) -> dict | None:
        try:
            lstm_meta_path = os.path.join(
                self.config.pushed_model_dir, "lstm_model_meta.json"
            )
            xgb_meta_path = os.path.join(
                self.config.pushed_model_dir, "xgb_model_meta.json"
            )

            if not os.path.exists(lstm_meta_path) or \
               not os.path.exists(xgb_meta_path):
                logger.info(
                    "No previous model found — "
                    "will use fixed thresholds."
                )
                return None

            with open(lstm_meta_path) as f:
                lstm_meta = json.load(f)
            with open(xgb_meta_path) as f:
                xgb_meta = json.load(f)

            prev = {
                "lstm_mae": lstm_meta["metrics"]["mae_celsius"],
                "lstm_rmse": lstm_meta["metrics"]["rmse_celsius"],
                "xgb_auc" : xgb_meta["roc_auc"],
                "xgb_f1" : xgb_meta["metrics_at_best_threshold"]["f1_score"],
            }
            logger.info(
                f"Previous model — "
                f"LSTM MAE: {prev['lstm_mae']} | "
                f"XGB AUC: {prev['xgb_auc']}"
            )
            return prev

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

            acc       = accuracy_score(y_test, y_pred)
            f1        = f1_score(y_test, y_pred, zero_division=0)
            auc_score = roc_auc_score(y_test, y_prob)

            # ROC plot
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
                f"F1: {f1:.4f} | ACC: {acc:.4f}"
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

            scaled = scaler.transform(
                df_model[features].values
            ).astype(np.float32)

            X_all, y_all = [], []
            for i in range(len(scaled) - lookback - forecast_n + 1):
                X_all.append(scaled[i: i + lookback])
                y_all.append(scaled[i + lookback: i + lookback + forecast_n, target_idx])
            X_all = np.array(X_all, dtype=np.float32)
            y_all = np.array(y_all, dtype=np.float32)

            val_end = int(len(X_all) * 0.85)
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
            ax.plot(range(n_show), y_true_C[:n_show, 0], label="Actual",   color="#E85D5D", lw=1.5)
            ax.plot(range(n_show), y_pred_C[:n_show, 0], label="Pred +1h", color="#4A90D9", lw=1.5, ls="--")
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

    # ── Acceptance logic ─────────────────────────────────────
    def _check_acceptance(
        self,
        lstm_results: dict,
        xgb_results: dict,
        previous: dict | None,
    ) -> tuple[bool, str]:

        new_mae = lstm_results["mae_celsius"]
        new_rmse = lstm_results["rmse_celsius"]
        new_auc = xgb_results["roc_auc"]
        new_f1 = xgb_results["f1_score"]

        if previous is None:
            # First run — use fixed thresholds
            lstm_ok = new_mae <= self.config.lstm_mae_threshold
            xgb_ok  = new_auc >= self.config.xgb_auc_threshold

            reason = (
                f"First run - fixed thresholds. "
                f"LSTM MAE {new_mae} <= {self.config.lstm_mae_threshold}: {lstm_ok} | "
                f"XGB AUC {new_auc} >= {self.config.xgb_auc_threshold}: {xgb_ok}"
            )
        else:
            # Compare against previous model
            # Primary must improve, secondary/tertiary can have small regression
            prev_mae = previous["lstm_mae"]
            prev_rmse = previous["lstm_rmse"]
            prev_auc = previous["xgb_auc"]
            prev_f1 = previous["xgb_f1"]
            
            lstm_ok = (new_mae < prev_mae) and (new_rmse < prev_rmse * 1.05)  # 5% tolerance on RMSE
            xgb_ok  = (new_auc > prev_auc) and (new_f1  > prev_f1  * 0.95)   # 5% tolerance on F1
            reason = (
                f"Compared to previous model. "
                f"LSTM MAE {new_mae} < {previous['lstm_mae']} (prev): {lstm_ok} | "
                f"XGB AUC {new_auc} > {previous['xgb_auc']} (prev): {xgb_ok}"
            )

        accepted = lstm_ok or xgb_ok
        logger.info(f"Acceptance check — {reason}")
        return accepted, reason

    # ── Orchestrator ─────────────────────────────────────────
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Evaluation")
            logger.info("=" * 60)

            # Load feature data
            path = os.path.join(DATA_TRANSFORMATION_FEATURES_DIR, "weather_final.csv")

            df = pd.read_csv(path, parse_dates=["valid_time"])
            df = df.sort_values("valid_time").reset_index(drop=True)

            create_directories([self.config.artifact_dir])

            # Load models
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

            # Evaluate
            xgb_results  = self._evaluate_xgboost(xgb_model, xgb_meta, df)
            lstm_results = self._evaluate_lstm(lstm_model, scaler, lstm_meta, df)

            # Load previous metrics and check acceptance
            previous              = self._load_previous_metrics()
            is_accepted, reason   = self._check_acceptance(
                lstm_results, xgb_results, previous
            )

            report = {
                "is_model_accepted": is_accepted,
                "acceptance_reason": reason,
                "new_model": {
                    "lstm_mae": lstm_results["mae_celsius"],
                    "lstm_rmse": lstm_results["rmse_celsius"],
                    "xgb_auc" : xgb_results["roc_auc"],
                    "xgb_f1" : xgb_results["f1_score"],
                },
                "previous_model": previous,
                "thresholds": {
                    "lstm_mae_max"  : self.config.lstm_mae_threshold,
                    "lstm_rmse_tol" : "5% tolerance",
                    "xgb_auc_min"   : self.config.xgb_auc_threshold,
                    "xgb_f1_tol"    : "5% tolerance",
                    "mode"          : "fixed" if previous is None else "comparison",
                },
                "xgboost" : xgb_results,
                "lstm"    : lstm_results,
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
                    "eval_xgb_roc_auc" : xgb_results["roc_auc"],
                    "eval_xgb_f1"      : xgb_results["f1_score"],
                    "eval_lstm_mae"    : lstm_results["mae_celsius"],
                    "eval_lstm_rmse"   : lstm_results["rmse_celsius"],
                    "eval_lstm_r2"     : lstm_results["r2_score"],
                    "model_accepted"   : int(is_accepted),
                })
                if previous:
                    mlflow.log_metrics({
                        "prev_lstm_mae": previous["lstm_mae"],
                        "prev_xgb_auc" : previous["xgb_auc"],
                        "lstm_mae_improvement": round(
                            previous["lstm_mae"] - lstm_results["mae_celsius"], 4
                        ),
                        "xgb_auc_improvement": round(
                            xgb_results["roc_auc"] - previous["xgb_auc"], 4
                        ),
                    })
                mlflow.log_artifact(self.config.report_file_path)
                try:
                    mlflow.log_artifact(xgb_results["roc_plot"])
                    mlflow.log_artifact(lstm_results["forecast_plot"])
                except Exception:
                    pass

            artifact = ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                lstm_mae=lstm_results["mae_celsius"],
                xgb_auc=xgb_results["roc_auc"],
                report_file_path=self.config.report_file_path,
                message=reason,
            )

            logger.info("=" * 60)
            logger.info(f"Model Evaluation Artifact: {artifact}")
            logger.info("=" * 60)
            return artifact

        except Exception as e:
            raise WeatherException(e, sys)

        
if __name__ == "__main__":
    import json
    from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
    from src.entity.artifact_entity import ModelTrainerArtifact

    # Load trainer artifact from the latest artifact directory
    # import glob
    # latest = sorted(glob.glob("artifact/*/model_trainer"))[-1]
    trainer_config = ModelTrainerConfig()
    trainer_artifact = ModelTrainerArtifact(
        lstm_model_path = trainer_config.lstm_model_path,
        xgb_model_path  = trainer_config.xgb_model_path,
        lstm_meta_path  = trainer_config.lstm_meta_path,
        xgb_meta_path   = trainer_config.xgb_meta_path,
        scaler_path     = trainer_config.scaler_path,
        lstm_mae        = 0.0,
        xgb_auc         = 0.0,
        is_trained      = True,
        message         = "Loaded from saved_models",
    )

    evaluator = ModelEvaluation(ModelEvaluationConfig(), trainer_artifact)
    artifact  = evaluator.initiate_model_evaluation()