"""
demo.py — Local development entry point

Usage:
    python demo.py pipeline    → run full training pipeline
    python demo.py predict     → run prediction pipeline
    python demo.py api         → start FastAPI server
    python demo.py             → start FastAPI server (default)
"""

import sys


def run_pipeline():
    from src.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()
    pipeline.run()


def run_prediction():
    from src.pipeline.prediction_pipeline import PredictionPipeline
    pipeline = PredictionPipeline()
    result   = pipeline.predict(
        latitude  = 26.00,
        longitude = 83.00,
        city      = "Azamgarh",
    )
    print(f"\nCity          : {result['city']}")
    print(f"Rain tomorrow : {result['rain_prediction']['rain_tomorrow']}")
    print(f"Probability   : {result['rain_prediction']['probability']}")
    print(f"Confidence    : {result['rain_prediction']['confidence']}")
    print(f"\nTemperature Forecast:")
    for item in result["temperature_forecast"]["forecast"]:
        print(f"  {item['hour']} → {item['temperature_C']}°C")


def run_api():
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = True,
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"

    if mode == "pipeline":
        run_pipeline()
    elif mode == "predict":
        run_prediction()
    elif mode == "api":
        run_api()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python demo.py [pipeline|predict|api]")
        sys.exit(1)