"""
XGBoost Trainer Module

Trains an XGBoost Classifier for next-day rain prediction:
1. Time-based train/test split (80/20, no shuffle).
2. 5-fold TimeSeriesSplit cross-validation.
3. Sample weight balancing for class imbalance.
4. Final model training with early stopping.
5. Threshold tuning to maximise F1-score.
6. Saves model and metadata to artifact directory.
7. Logs all params and metrics to MLflow.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, classification_report,
)

import mlflow

from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import read_yaml_file, save_json, create_directories


class XGBoostTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config   = config
        self.params   = read_yaml_file("params.yaml")
        self.xgb_cfg  = self.params["xgboost"]
        self.split    = self.params["split"]
        self.features = self.xgb_cfg["features"]
        self.target   = "rain_tomorrow"

    # ── Threshold tuning ─────────────────────────────────────
    def _find_best_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> float:
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.10, 0.91, 0.05):
            f1 = f1_score(
                y_true,
                (y_prob >= thr).astype(int),
                zero_division=0,
            )
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        return round(float(best_thr), 2)
    
    # ── Cross-validation ─────────────────────────────────────
    def _cross_validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        xgb_params: dict,
    ) -> dict:

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {"accuracy": [], "f1": [], "roc_auc": []}

        for fold, (tr_idx, val_idx) in enumerate(
            tscv.split(X_train), 1
        ):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            # Balance classes using only the fold's training data.
            sw = compute_sample_weight("balanced", y=ytr)

            m = XGBClassifier(**xgb_params)

            m.fit(
                Xtr,
                ytr,
                sample_weight=sw,
                eval_set=[(Xval, yval)],
                verbose=False,
            )

            yp = m.predict(Xval)
            ypp = m.predict_proba(Xval)[:, 1]

            cv_scores["accuracy"].append(
                accuracy_score(yval, yp)
            )
            cv_scores["f1"].append(
                f1_score(yval, yp, zero_division=0)
            )
            cv_scores["roc_auc"].append(
                roc_auc_score(yval, ypp)
            )

            logger.info(
                f"Fold {fold} - "
                f"ACC: {cv_scores['accuracy'][-1]:.4f}  "
                f"F1: {cv_scores['f1'][-1]:.4f}  "
                f"AUC: {cv_scores['roc_auc'][-1]:.4f}"
            )

        return cv_scores


    # ── Main train ───────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        try:
            logger.info("-" * 60)
            logger.info("XGBoost — Rain Prediction Classifier")
            logger.info("-" * 60)

            # Validate columns
            missing = [
                c for c in self.features + [self.target]
                if c not in df.columns
            ]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            df = df.sort_values("valid_time").reset_index(drop=True)
            X  = df[self.features]
            y  = df[self.target].astype(int)

            logger.info(
                f"Dataset: {df.shape} | "
                f"Rain=1: {y.sum():,} ({100*y.mean():.1f}%) | "
                f"Rain=0: {(~y.astype(bool)).sum():,} ({100*(1-y.mean()):.1f}%)"
            )

            # Time-based split
            split_idx       = int(len(df) * (1 - self.split["test_size"]))
            X_train, X_test = X.iloc[:split_idx],  X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx],  y.iloc[split_idx:]

            logger.info(
                f"Train: {len(X_train):,} | Test: {len(X_test):,}"
            )

            sample_weights = compute_sample_weight("balanced", y=y_train)

            xgb_params = {
                "n_estimators":          self.xgb_cfg["n_estimators"],
                "max_depth":             self.xgb_cfg["max_depth"],
                "learning_rate":         self.xgb_cfg["learning_rate"],
                "subsample":             self.xgb_cfg["subsample"],
                "colsample_bytree":      self.xgb_cfg["colsample_bytree"],
                "min_child_weight":      self.xgb_cfg["min_child_weight"],
                "gamma":                 self.xgb_cfg["gamma"],
                "reg_alpha":             self.xgb_cfg["reg_alpha"],
                "reg_lambda":            self.xgb_cfg["reg_lambda"],
                "use_label_encoder":     self.xgb_cfg["use_label_encoder"],
                "eval_metric":           self.xgb_cfg["eval_metric"],
                "early_stopping_rounds": self.xgb_cfg["early_stopping_rounds"],
                "random_state":          self.xgb_cfg["random_state"],
                "n_jobs":                self.xgb_cfg["n_jobs"],
            }

            # Cross-validation
            logger.info("5-fold TimeSeriesSplit cross-validation...")
            cv_scores = self._cross_validate(X_train, y_train, xgb_params)

            logger.info(
                f"CV — AUC: {np.mean(cv_scores['roc_auc']):.4f} "
                f"± {np.std(cv_scores['roc_auc']):.4f}"
            )

            # Final model
            logger.info("Training final model...")
            model = XGBClassifier(
                **xgb_params,
            )
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
            logger.info(f"Best iteration: {model.best_iteration}")

            # Evaluate at 0.5
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            f1     = f1_score(y_test, y_pred, zero_division=0)
            auc    = roc_auc_score(y_test, y_prob)
            ap     = average_precision_score(y_test, y_prob)

            logger.info(
                f"Metrics @0.5 — ACC: {acc:.4f} F1: {f1:.4f} "
                f"AUC: {auc:.4f} AP: {ap:.4f}"
            )
            logger.info(
                "\n" + classification_report(
                    y_test, y_pred,
                    target_names=["No Rain", "Rain"],
                    zero_division=0,
                )
            )

            # Threshold tuning
            best_thr     = self._find_best_threshold(y_test.values, y_prob)
            y_pred_tuned = (y_prob >= best_thr).astype(int)
            f1_tuned     = f1_score(y_test, y_pred_tuned, zero_division=0)
            acc_tuned    = accuracy_score(y_test, y_pred_tuned)

            logger.info(
                f"Tuned @{best_thr} — ACC: {acc_tuned:.4f} F1: {f1_tuned:.4f}"
            )

            # Feature importance
            importance = model.get_booster().get_score(importance_type="gain")
            importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )

            meta = {
                "model":           "XGBoostClassifier",
                "features":        self.features,
                "target":          self.target,
                "best_threshold":  best_thr,
                "best_iteration":  int(model.best_iteration),
                "train_rows":      int(len(X_train)),
                "test_rows":       int(len(X_test)),
                "hyperparameters": xgb_params,
                "cv_results": {
                    "mean_accuracy": round(float(np.mean(cv_scores["accuracy"])), 4),
                    "std_accuracy":  round(float(np.std(cv_scores["accuracy"])),  4),
                    "mean_f1":       round(float(np.mean(cv_scores["f1"])),       4),
                    "std_f1":        round(float(np.std(cv_scores["f1"])),        4),
                    "mean_roc_auc":  round(float(np.mean(cv_scores["roc_auc"])),  4),
                    "std_roc_auc":   round(float(np.std(cv_scores["roc_auc"])),   4),
                },
                "roc_auc":        round(float(auc), 4),
                "avg_precision":  round(float(ap),  4),
                "metrics_at_0_5": {
                    "accuracy": round(float(acc), 4),
                    "f1_score": round(float(f1),  4),
                },
                "metrics_at_best_threshold": {
                    "threshold": best_thr,
                    "accuracy":  round(float(acc_tuned), 4),
                    "f1_score":  round(float(f1_tuned),  4),
                },
                "feature_importance_gain": {
                    k: round(v, 4) for k, v in importance.items()
                },
            }

            # Save artifacts
            create_directories([os.path.dirname(self.config.xgb_model_path)])
            model.save_model(self.config.xgb_model_path)
            save_json(self.config.xgb_meta_path, meta)
            logger.info(f"XGBoost model -> {self.config.xgb_model_path}")
            logger.info(f"XGBoost meta  -> {self.config.xgb_meta_path}")

            # MLflow logging
            mlflow.log_params(xgb_params)
            mlflow.log_params({"best_threshold": best_thr})
            mlflow.log_metrics({
                "roc_auc":       round(float(auc), 4),
                "f1_tuned":      round(float(f1_tuned), 4),
                "cv_mean_auc":   round(float(np.mean(cv_scores["roc_auc"])), 4),
                "avg_precision": round(float(ap), 4),
            })
            mlflow.log_artifact(self.config.xgb_model_path)
            mlflow.log_artifact(self.config.xgb_meta_path)

            logger.info("-" * 60)
            logger.info(
                f"XGBoost complete - "
                f"AUC: {auc:.4f} | F1 tuned: {f1_tuned:.4f} @ {best_thr}"
            )
            logger.info("-" * 60)

            return {
                "model": model,
                "meta":  meta,
                "auc":   auc,
            }

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig, DataTransformationConfig,
        DataValidationConfig, ModelTrainerConfig,
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    from src.components.data_validation import DataValidation

    ingestion_artifact      = DataIngestion(DataIngestionConfig()).initiate_data_ingestion()
    transformation_artifact = DataTransformation(DataTransformationConfig(), ingestion_artifact).initiate_data_transformation()
    validation_artifact     = DataValidation(DataValidationConfig(), transformation_artifact).initiate_data_validation()

    config  = ModelTrainerConfig()
    trainer = XGBoostTrainer(config)

    df      = pd.read_csv(
        transformation_artifact.final_features_path,
        parse_dates=["valid_time"],
    )
    
    mlflow.set_experiment("weather_rain_xgboost_pipeline")

    with mlflow.start_run(run_name="xgboost_rain_training"):
        result = trainer.train(df)