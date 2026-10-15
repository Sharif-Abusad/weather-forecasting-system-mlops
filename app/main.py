"""
app/main.py — FastAPI Application Entry Point

Handles:
1. App initialization with lifespan
2. CORS middleware
3. Route registration
4. Static file serving
5. Frontend delivery
6. Health check
7. Model metadata endpoint
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routes.weather_routes import router
from app.services.predictor import load_models, get_loaded_meta


# ── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Weather Forecasting API...")
    load_models()
    print("Models loaded and ready")
    yield
    print("Shutting down...")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title       = "Weather Forecasting API",
    description = (
        "XGBoost rain prediction + "
        "BiLSTM temperature forecasting — Azamgarh, UP"
    ),
    version  = "2.0.0",
    lifespan = lifespan,
    docs_url = "/docs",
    redoc_url= "/redoc",
)


# ── CORS ──────────────────────────────────────────────────
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Routes ────────────────────────────────────────────────
app.include_router(router)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Health check ──────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status"       : "ok",
        "models_loaded": True,
        "version"      : "2.0.0",
    }


# ── Model metadata ────────────────────────────────────────
@app.get("/model-meta", tags=["Models"])
def get_model_meta():
    """
    Returns metadata for both models — served from
    already-loaded in-memory dicts (no file reads).
    """
    try:
        return get_loaded_meta()
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"detail": str(e)},
        )


# ── Frontend ──────────────────────────────────────────────
@app.get("/", tags=["Frontend"])
def serve_ui():
    index_path = Path("templates/index.html")
    if not index_path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "index.html not found"},
        )
    return FileResponse(str(index_path))


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host  = "0.0.0.0",
        port  = int(os.getenv("PORT", 8000)),
        reload= os.getenv("DEBUG", "1") == "1",
    )