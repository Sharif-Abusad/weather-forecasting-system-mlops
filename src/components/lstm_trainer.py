"""
LSTM Trainer Module

Trains an Encoder-Decoder BiLSTM for multi-step temperature forecasting:
1. Loads feature dataset and selects LSTM feature columns.
2. Fits MinMaxScaler on full dataset.
3. Generates sliding window sequences (lookback=24h → forecast=5h).
4. Time-based 70/15/15 train/val/test split (no shuffle).
5. Builds Encoder-Decoder BiLSTM architecture.
6. Trains with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
7. Evaluates overall and per-step MAE/RMSE/R².
8. Saves model (.keras), scaler (.pkl) and metadata (.json).
9. Logs all params, metrics and artifacts to MLflow.
"""

import os
import sys
import json
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_DETERMINISTIC_OPS"]  = "1"
os.environ["PYTHONHASHSEED"]         = "42"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import joblib
import mlflow
import mlflow.keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    RepeatVector, TimeDistributed, Reshape,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import read_yaml_file, save_json, create_directories


class LSTMTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config   = config
        self.params   = read_yaml_file("params.yaml")
        self.lstm_cfg = self.params["lstm"]
        self.features = self.lstm_cfg["features"]
        self.target_idx = 0   # temperature is always index 0

    # ── Sequence generator ───────────────────────────────────
    def _create_sequences(
        self,
        data: np.ndarray,
        lookback: int,
        forecast_n: int,
    ):
        X, y = [], []
        for i in range(len(data) - lookback - forecast_n + 1):
            X.append(data[i: i + lookback])
            y.append(data[i + lookback: i + lookback + forecast_n, self.target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ── Inverse transform temperature only ──────────────────
    def _inverse_temp(
        self,
        scaled_seq: np.ndarray,
        scaler: MinMaxScaler,
        n_features: int,
    ) -> np.ndarray:
        dummy = np.zeros((len(scaled_seq), n_features), dtype=np.float32)
        dummy[:, self.target_idx] = scaled_seq
        return scaler.inverse_transform(dummy)[:, self.target_idx]

    # ── Model builder ────────────────────────────────────────
    def _build_model(
        self,
        lookback: int,
        forecast_n: int,
        n_features: int,
        units: int,
        dropout: float,
        learning_rate: float,
    ) -> Model:
        kernel_init    = GlorotUniform(seed=SEED)
        recurrent_init = Orthogonal(gain=1.0, seed=SEED)

        inputs = Input(shape=(lookback, n_features), name="encoder_input")

        x = Bidirectional(
            LSTM(
                units,
                return_sequences=True,
                kernel_initializer=kernel_init,
                recurrent_initializer=recurrent_init,
                dropout=dropout * 0.5,
                recurrent_dropout=0.0,
            ),
            name="bilstm_encoder",
        )(inputs)
        x = Dropout(dropout, seed=SEED, name="dropout_enc")(x)
        x = LSTM(
            units // 2,
            return_sequences=False,
            kernel_initializer=kernel_init,
            recurrent_initializer=recurrent_init,
            dropout=dropout * 0.5,
            name="lstm_context",
        )(x)
        x = RepeatVector(forecast_n, name="repeat_vector")(x)
        x = LSTM(
            units // 2,
            return_sequences=True,
            kernel_initializer=kernel_init,
            recurrent_initializer=recurrent_init,
            dropout=dropout * 0.5,
            name="lstm_decoder",
        )(x)
        x = Dropout(dropout, seed=SEED, name="dropout_dec")(x)
        x = TimeDistributed(Dense(32, activation="relu"), name="td_dense")(x)
        x = TimeDistributed(Dense(1),                    name="td_output")(x)
        x = Reshape((forecast_n,),                       name="output")(x)

        model = Model(inputs, x, name="BiLSTM_TempForecaster")
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="huber",
            metrics=["mae"],
        )
        return model

    # ── Main train ───────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        try:
            logger.info("-" * 60)
            logger.info("LSTM — Temperature Forecaster")
            logger.info("-" * 60)

            lookback      = self.lstm_cfg["lookback"]
            forecast_n    = self.lstm_cfg["forecast_n"]
            units         = self.lstm_cfg["units"]
            dropout       = self.lstm_cfg["dropout"]
            learning_rate = self.lstm_cfg.get("learning_rate", 5e-4)
            epochs        = self.lstm_cfg["epochs"]
            batch_size    = self.lstm_cfg["batch_size"]
            patience      = self.lstm_cfg["patience"]

            missing = [c for c in self.features if c not in df.columns]
            if missing:
                raise ValueError(f"Missing LSTM features: {missing}")

            df_model   = (
                df[["valid_time"] + self.features]
                .dropna()
                .sort_values("valid_time")
                .reset_index(drop=True)
            )
            n_features = len(self.features)

            logger.info(
                f"Rows: {len(df_model):,} | "
                f"Features: {n_features} | "
                f"Lookback: {lookback}h | "
                f"Forecast: {forecast_n}h"
            )

            # Scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(
                df_model[self.features].values
            ).astype(np.float32)

            # Sequences
            X, y = self._create_sequences(scaled, lookback, forecast_n)
            logger.info(f"Sequences — X: {X.shape} | y: {y.shape}")

            # 70/15/15 split
            n         = len(X)
            train_end = int(n * 0.70)
            val_end   = int(n * 0.85)
            X_train, y_train = X[:train_end],        y[:train_end]
            X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
            X_test,  y_test  = X[val_end:],          y[val_end:]

            logger.info(
                f"Split — Train: {len(X_train):,} | "
                f"Val: {len(X_val):,} | "
                f"Test: {len(X_test):,}"
            )

            # Build model
            model = self._build_model(
                lookback, forecast_n, n_features,
                units, dropout, learning_rate,
            )
            logger.info(f"Model params: {model.count_params():,}")

            # Callbacks
            create_directories([self.config.artifact_dir])
            ckpt_path = os.path.join(self.config.artifact_dir, "best_lstm.keras")

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=1e-4,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1,
                    cooldown=3,
                ),
                ModelCheckpoint(
                    filepath=ckpt_path,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=0,
                ),
            ]

            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                shuffle=False,
                verbose=1,
            )

            best_epoch    = int(np.argmin(history.history["val_loss"])) + 1
            best_val_loss = float(np.min(history.history["val_loss"]))
            logger.info(
                f"Best epoch: {best_epoch} | val_loss: {best_val_loss:.6f}"
            )

            # Evaluate
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_true_C = np.array([
                self._inverse_temp(y_test[i],        scaler, n_features)
                for i in range(len(y_test))
            ])
            y_pred_C = np.array([
                self._inverse_temp(y_pred_scaled[i], scaler, n_features)
                for i in range(len(y_pred_scaled))
            ])

            mae  = mean_absolute_error(y_true_C.flatten(), y_pred_C.flatten())
            rmse = np.sqrt(mean_squared_error(y_true_C.flatten(), y_pred_C.flatten()))
            r2   = r2_score(y_true_C.flatten(), y_pred_C.flatten())

            per_step = {}
            for step in range(forecast_n):
                s_mae  = mean_absolute_error(y_true_C[:, step], y_pred_C[:, step])
                s_rmse = np.sqrt(mean_squared_error(y_true_C[:, step], y_pred_C[:, step]))
                s_r2   = r2_score(y_true_C[:, step], y_pred_C[:, step])
                per_step[f"+{step+1}h"] = {
                    "mae":  round(float(s_mae),  4),
                    "rmse": round(float(s_rmse), 4),
                    "r2":   round(float(s_r2),   4),
                }
                logger.info(
                    f"  +{step+1}h — MAE: {s_mae:.4f}°C | "
                    f"RMSE: {s_rmse:.4f}°C | R²: {s_r2:.4f}"
                )

            logger.info(
                f"Overall — MAE: {mae:.4f}°C | "
                f"RMSE: {rmse:.4f}°C | R²: {r2:.4f}"
            )

            # Save
            create_directories([
                os.path.dirname(self.config.lstm_model_path),
                os.path.dirname(self.config.scaler_path),
            ])

            model.save(self.config.lstm_model_path)
            joblib.dump(scaler, self.config.scaler_path)

            meta = {
                "model"         : "Encoder-Decoder BiLSTM",
                "target"        : "temperature",
                "features"      : self.features,
                "target_idx"    : self.target_idx,
                "lookback"      : lookback,
                "forecast_n"    : forecast_n,
                "n_features"    : n_features,
                "hyperparameters": {
                    "lstm_units"   : units,
                    "dropout"      : dropout,
                    "learning_rate": learning_rate,
                    "batch_size"   : batch_size,
                },
                "best_epoch"    : best_epoch,
                "best_val_loss" : round(best_val_loss, 6),
                "train_samples" : int(len(X_train)),
                "val_samples"   : int(len(X_val)),
                "test_samples"  : int(len(X_test)),
                "metrics": {
                    "mae_celsius"  : round(float(mae),  4),
                    "rmse_celsius" : round(float(rmse), 4),
                    "r2_score"     : round(float(r2),   4),
                },
                "per_step_metrics": per_step,
            }

            save_json(self.config.lstm_meta_path, meta)

            logger.info(f"LSTM model  -> {self.config.lstm_model_path}")
            logger.info(f"LSTM scaler -> {self.config.scaler_path}")
            logger.info(f"LSTM meta   -> {self.config.lstm_meta_path}")

            # MLflow
            mlflow.log_params({
                "lookback"     : lookback,
                "forecast_n"   : forecast_n,
                "lstm_units"   : units,
                "dropout"      : dropout,
                "learning_rate": learning_rate,
                "batch_size"   : batch_size,
            })
            mlflow.log_metrics({
                "mae_celsius"  : round(float(mae),        4),
                "rmse_celsius" : round(float(rmse),       4),
                "r2_score"     : round(float(r2),         4),
                "best_epoch"   : best_epoch,
                "best_val_loss": round(best_val_loss,     6),
            })

            for step_name, step_metrics in per_step.items():
                clean = step_name.replace("+", "plus_").replace("h", "hr")  # +1h → plus_1hr
                mlflow.log_metric(f"mae_{clean}",  step_metrics["mae"])
                mlflow.log_metric(f"rmse_{clean}", step_metrics["rmse"])
                mlflow.log_metric(f"r2_{clean}",   step_metrics["r2"])

            mlflow.log_artifact(self.config.lstm_model_path)
            mlflow.log_artifact(self.config.lstm_meta_path)
            mlflow.keras.log_model(model, "lstm_model")

            logger.info("-" * 60)
            logger.info(
                f"LSTM complete — MAE: {mae:.4f}°C | "
                f"RMSE: {rmse:.4f}°C | R²: {r2:.4f}"
            )
            logger.info("-" * 60)

            return {
                "model"  : model,
                "scaler" : scaler,
                "meta"   : meta,
                "mae"    : mae,
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

    df     = pd.read_csv(
        transformation_artifact.final_features_path,
        parse_dates=["valid_time"],
    )
    config = ModelTrainerConfig()

    mlflow.set_experiment("weather_temperature_lstm_pipeline")
    with mlflow.start_run(run_name="lstm_temperature"):
        result = LSTMTrainer(config).train(df)