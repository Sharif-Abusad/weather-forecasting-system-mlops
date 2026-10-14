"""
Weather API Routes
────────────────────────────────────────────────────────────
Endpoints:
    GET /weather/{city}          — full dashboard data
    GET /weather/{city}/rain     — rain prediction only
    GET /weather/{city}/forecast — temperature forecast only
"""

from datetime import datetime, timedelta

import pytz
from fastapi import APIRouter, HTTPException

from app.services.predictor import predict_rain, predict_temperature
from app.services.weather_service import fetch_current_weather
from app.services.forecast_service import get_forecast
from app.utils.weather_utils import validate_and_geocode

router = APIRouter()


# ── Helper ────────────────────────────────────────────────
def _get_coords_and_weather(city: str):
    """Shared logic — geocode + fetch current weather."""
    latitude, longitude, display_name = validate_and_geocode(city)
    weather_data = fetch_current_weather(latitude, longitude)
    weather_data["current_weather"]["city"] = display_name
    return latitude, longitude, display_name, weather_data


# ── 1. Full dashboard endpoint ────────────────────────────
@router.get("/weather/{city}", tags=["Weather"])
def weather(city: str):
    """
    Full weather dashboard data.
    Returns current conditions, LSTM hourly forecast,
    XGBoost rain prediction and 5-day forecast.
    """
    try:
        latitude, longitude, display_name, weather_data = \
            _get_coords_and_weather(city)

        lstm_forecast   = predict_temperature(weather_data["hourly_df"])
        daily_forecast  = get_forecast(latitude, longitude)
        rain_prediction = predict_rain(weather_data["xgboost_features"])

        timezone     = pytz.timezone("Asia/Kolkata")
        current_time = datetime.now(timezone)

        hourly_forecast = [
            {
                "time"    : (current_time + timedelta(hours=i + 1)).strftime("%H:00"),
                "temp"    : round(item["temperature_C"]),
                "humidity": None,
            }
            for i, item in enumerate(lstm_forecast["forecast"])
        ]

        return {
            "current"        : weather_data["current_weather"],
            "hourly"         : hourly_forecast,
            "forecast"       : daily_forecast,
            "rain_prediction": rain_prediction,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. Rain prediction only ───────────────────────────────
@router.get("/weather/{city}/rain", tags=["Predictions"])
def rain_prediction(city: str):
    """
    XGBoost rain prediction only.
    Returns rain_tomorrow, probability and confidence.
    """
    try:
        _, _, _, weather_data = _get_coords_and_weather(city)
        result = predict_rain(weather_data["xgboost_features"])

        return {
            "city"           : weather_data["current_weather"]["city"],
            "rain_prediction": result,
            "features_used"  : list(weather_data["xgboost_features"].keys()),
            "timestamp"      : datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Temperature forecast only ─────────────────────────
@router.get("/weather/{city}/forecast", tags=["Predictions"])
def temperature_forecast(city: str):
    """
    LSTM temperature forecast only.
    Returns next 5-hour temperature predictions with MAE.
    """
    try:
        _, _, _, weather_data = _get_coords_and_weather(city)
        result = predict_temperature(weather_data["hourly_df"])

        timezone     = pytz.timezone("Asia/Kolkata")
        current_time = datetime.now(timezone)

        hourly = [
            {
                "time"         : (current_time + timedelta(hours=i + 1)).strftime("%H:00"),
                "hour"         : item["hour"],
                "temperature_C": item["temperature_C"],
            }
            for i, item in enumerate(result["forecast"])
        ]

        return {
            "city"       : weather_data["current_weather"]["city"],
            "forecast"   : hourly,
            "n_hours"    : result["n_hours"],
            "mae_celsius": result["mae_celsius"],
            "model"      : "Encoder-Decoder BiLSTM",
            "timestamp"  : datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))