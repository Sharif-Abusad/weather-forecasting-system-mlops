import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Location ──────────────────────────────────────────────
LAT       = float(os.getenv("LATITUDE",  26.00))
LON       = float(os.getenv("LONGITUDE", 83.00))
TIMEZONE  = os.getenv("TIMEZONE", "Asia/Kolkata")

# ── Model settings ────────────────────────────────────────
FORECAST_N    = int(os.getenv("FORECAST_N",    5))
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", 5))
LOOKBACK      = int(os.getenv("LOOKBACK",      24))

# ── Paths ─────────────────────────────────────────────────
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")

# ── Model directory (single source of truth) ──────────────
# Training pipeline pushes all models here via model_pusher
PUSHED_MODEL_DIR = os.getenv("PUSHED_MODEL_DIR", "deployment")

# ── LSTM paths ────────────────────────────────────────────
MODEL_LSTM_DIR   = PUSHED_MODEL_DIR
LSTM_MODEL_FILE  = os.getenv("LSTM_MODEL_FILE", "lstm_temp_model.keras")
LSTM_META_FILE   = os.getenv("LSTM_META_FILE",  "lstm_model_meta.json")
MODEL_LSTM       = os.path.join(PUSHED_MODEL_DIR, LSTM_MODEL_FILE)
LSTM_META        = os.path.join(PUSHED_MODEL_DIR, LSTM_META_FILE)

# ── Scaler paths ──────────────────────────────────────────
SCALER_DIR   = PUSHED_MODEL_DIR
SCALER_FILE  = os.getenv("SCALER_FILE", "lstm_temp_scaler.pkl")
SCALER_PATH  = os.path.join(PUSHED_MODEL_DIR, SCALER_FILE)

# ── XGBoost paths ─────────────────────────────────────────
MODEL_XGBOOST_DIR  = PUSHED_MODEL_DIR
XGBOOST_MODEL_FILE = os.getenv("XGBOOST_MODEL_FILE", "rain_model.json")
XGBOOST_META_FILE  = os.getenv("XGBOOST_META_FILE",  "xgb_model_meta.json")
MODEL_XGBOOST      = os.path.join(PUSHED_MODEL_DIR, XGBOOST_MODEL_FILE)
XGBOOST_META       = os.path.join(PUSHED_MODEL_DIR, XGBOOST_META_FILE)

# ── Open-Meteo ────────────────────────────────────────────
OPEN_METEO_URL = os.getenv(
    "OPEN_METEO_BASE_URL",
    "https://api.open-meteo.com/v1/forecast"
)

# ── Geocoding ─────────────────────────────────────────────
GEOCODE_URL = os.getenv(
    "GEOCODE_URL",
    "https://geocoding-api.open-meteo.com/v1/search"
)

# ── FastAPI ───────────────────────────────────────────────
PORT  = int(os.getenv("APP_PORT", 8000))
DEBUG = os.getenv("DEBUG", "1") == "1"