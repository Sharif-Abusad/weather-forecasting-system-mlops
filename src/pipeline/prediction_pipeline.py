"""
Prediction Pipeline

Serves live predictions using models from saved_models/:
1. Loads LSTM model, scaler and metadata.
2. Loads XGBoost model and metadata.
3. Fetches last 24h of live weather data from Open-Meteo.
4. Predicts next 5h temperature using LSTM.
5. Predicts rain tomorrow using XGBoost.
6. Returns structured prediction result.
"""

import os
import sys
import json
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from xgboost import XGBClassifier

from src.logger import logger
from src.exception import WeatherException
from src.constants import (
    PUSHED_MODEL_DIR,
    LSTM_MODEL_FILE,
    XGB_MODEL_FILE,
    LSTM_META_FILE,
    XGB_META_FILE,
    SCALER_FILE_NAME,
    DEFAULT_TIMEZONE,
)


class PredictionPipeline:

    def __init__(self):
        self.pushed_model_dir = PUSHED_MODEL_DIR
        self._lstm_model     = None
        self._xgb_model      = None
        self._scaler         = None
        self._lstm_meta      = None
        self._xgb_meta       = None

    # ── Lazy load models ─────────────────────────────────────
    def _load_models(self):
        try:
            if self._lstm_model is not None:
                return

            lstm_path  = os.path.join(self.pushed_model_dir, LSTM_MODEL_FILE)
            xgb_path   = os.path.join(self.pushed_model_dir, XGB_MODEL_FILE)
            scaler_path = os.path.join(self.pushed_model_dir, SCALER_FILE_NAME)
            lstm_meta_path = os.path.join(self.pushed_model_dir, LSTM_META_FILE)
            xgb_meta_path  = os.path.join(self.pushed_model_dir, XGB_META_FILE)

            for path in [lstm_path, xgb_path, scaler_path, lstm_meta_path, xgb_meta_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Model file not found: {path}. "
                        f"Run training pipeline first."
                    )

            self._lstm_model = tf.keras.models.load_model(
                lstm_path, safe_mode=False
            )
            self._xgb_model = XGBClassifier()
            self._xgb_model.load_model(xgb_path)
            self._scaler = joblib.load(scaler_path)

            with open(lstm_meta_path) as f:
                self._lstm_meta = json.load(f)
            with open(xgb_meta_path) as f:
                self._xgb_meta = json.load(f)

            logger.info("Prediction models loaded successfully")

        except Exception as e:
            raise WeatherException(e, sys)

    # ── Fetch live weather ────────────────────────────────────
    def _fetch_live_weather(
        self,
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude"  : latitude,
                "longitude" : longitude,
                "hourly"    : [
                    "temperature_2m",
                    "dewpoint_2m",
                    "surface_pressure",
                    "cloud_cover",
                    "cloud_cover_low",
                    "cloud_cover_mid",
                    "cloud_cover_high",
                    "precipitation",
                    "wind_speed_10m",
                ],
                "past_hours"    : 24,
                "forecast_hours": 1,
                "timezone"      : DEFAULT_TIMEZONE,
            }

            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()["hourly"]

            dew  = np.array(data["dewpoint_2m"])
            temp = np.array(data["temperature_2m"])
            humidity = 100 * (
                np.exp((17.625 * dew)  / (243.04 + dew)) /
                np.exp((17.625 * temp) / (243.04 + temp))
            )

            df = pd.DataFrame({
                "valid_time"       : pd.to_datetime(data["time"]),
                "temperature"      : data["temperature_2m"],
                "surface_pressure" : data["surface_pressure"],
                "total_cloud_cover": [x / 100 for x in data["cloud_cover"]],
                "low_cloud_cover"  : [x / 100 for x in data["cloud_cover_low"]],
                "medium_cloud_cover":[x / 100 for x in data["cloud_cover_mid"]],
                "high_cloud_cover" : [x / 100 for x in data["cloud_cover_high"]],
                "precipitation"    : data["precipitation"],
                "humidity"         : humidity,
                "wind_speed"       : data["wind_speed_10m"],
            })

            # Add engineered features
            df["month"]          = df["valid_time"].dt.month
            df["hour"]           = df["valid_time"].dt.hour
            df["temp_lag_1"]     = df["temperature"].shift(1)
            df["temp_lag_24"]    = df["temperature"].shift(24)
            df["temp_rolling_6"] = df["temperature"].rolling(6, min_periods=1).mean()
            df["hour_sin"]       = np.sin(2 * np.pi * df["hour"]  / 24)
            df["hour_cos"]       = np.cos(2 * np.pi * df["hour"]  / 24)
            df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
            df = df.ffill().bfill()

            logger.info(f"Live weather fetched — {len(df)} hourly rows")
            return df

        except Exception as e:
            raise WeatherException(e, sys)

    # ── LSTM temperature forecast ─────────────────────────────
    def _predict_temperature(self, df: pd.DataFrame) -> dict:
        try:
            meta       = self._lstm_meta
            features   = meta["features"]
            lookback   = meta["lookback"]
            forecast_n = meta["forecast_n"]
            n_features = meta["n_features"]
            target_idx = meta["target_idx"]

            assert len(df) >= lookback, \
                f"Need {lookback} rows, got {len(df)}"

            window = df[features].iloc[-lookback:].values
            window_scaled = self._scaler.transform(window).reshape(
                1, lookback, n_features
            )

            pred_scaled = self._lstm_model.predict(
                window_scaled, verbose=0
            )[0][:forecast_n]

            dummy = np.zeros((forecast_n, n_features))
            dummy[:, target_idx] = pred_scaled
            pred_temps = self._scaler.inverse_transform(dummy)[:, target_idx]

            forecast = [
                {
                    "hour"         : f"+{i+1}h",
                    "temperature_C": round(float(t), 2),
                }
                for i, t in enumerate(pred_temps)
            ]

            return {
                "forecast"   : forecast,
                "n_hours"    : forecast_n,
                "mae_celsius": meta["metrics"]["mae_celsius"],
                "model"      : meta["model"],
            }

        except Exception as e:
            raise WeatherException(e, sys)

    # ── XGBoost rain prediction ───────────────────────────────
    def _predict_rain(self, df: pd.DataFrame) -> dict:
        try:
            meta      = self._xgb_meta
            features  = meta["features"]
            threshold = meta["best_threshold"]

            row  = df[features].iloc[-1:].copy()
            prob = float(self._xgb_model.predict_proba(row)[0][1])
            will_rain = prob >= threshold

            if prob > 0.75:   confidence = "high (rain likely)"
            elif prob > 0.55: confidence = "medium"
            elif prob < 0.25: confidence = "high (no rain likely)"
            else:             confidence = "low (uncertain)"

            return {
                "rain_tomorrow" : bool(will_rain),
                "probability"   : round(prob, 4),
                "confidence"    : confidence,
                "threshold_used": threshold,
            }

        except Exception as e:
            raise WeatherException(e, sys)

    # ── Main predict ─────────────────────────────────────────
    def predict(
        self,
        latitude: float,
        longitude: float,
        city: str = "",
    ) -> dict:
        try:
            self._load_models()

            df            = self._fetch_live_weather(latitude, longitude)
            temp_forecast = self._predict_temperature(df)
            rain_forecast = self._predict_rain(df)

            current = df.iloc[-1]

            result = {
                "city"            : city,
                "latitude"        : latitude,
                "longitude"       : longitude,
                "current"         : {
                    "temperature"      : round(float(current["temperature"]), 2),
                    "humidity"         : round(float(current["humidity"]),    2),
                    "surface_pressure" : round(float(current["surface_pressure"]), 2),
                    "precipitation"    : round(float(current["precipitation"]),    4),
                    "wind_speed"       : round(float(current["wind_speed"]),       2),
                    "total_cloud_cover": round(float(current["total_cloud_cover"]),4),
                    "low_cloud_cover"  : round(float(current["low_cloud_cover"]),  4),
                    "medium_cloud_cover":round(float(current["medium_cloud_cover"]),4),
                    "high_cloud_cover" : round(float(current["high_cloud_cover"]), 4),
                    "temp_lag_1"       : round(float(current["temp_lag_1"]),       2),
                    "temp_lag_24"      : round(float(current["temp_lag_24"]),      2),
                    "temp_rolling_6"   : round(float(current["temp_rolling_6"]),   2),
                    "month"            : int(current["month"]),
                },
                "temperature_forecast": temp_forecast,
                "rain_prediction"     : rain_forecast,
                "timestamp"           : datetime.now().isoformat(),
            }

            logger.info(
                f"Prediction complete — "
                f"City: {city} | "
                f"Rain: {rain_forecast['rain_tomorrow']} | "
                f"LSTM MAE: ±{temp_forecast['mae_celsius']}°C"
            )

            return result

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    result   = pipeline.predict(
        latitude  = 26.00,
        longitude = 83.00,
        city      = "Azamgarh",
    )
    print(f"Rain tomorrow : {result['rain_prediction']['rain_tomorrow']}")
    print(f"Probability   : {result['rain_prediction']['probability']}")
    print(f"Confidence    : {result['rain_prediction']['confidence']}")
    print(f"\nTemperature Forecast:")
    for item in result["temperature_forecast"]["forecast"]:
        print(f"  {item['hour']} → {item['temperature_C']}°C")