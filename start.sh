#!/bin/sh
set -e

cd /app

if [ ! -d .git ]; then
    git init
fi

echo "Configuring DVC authentication..."

dvc remote modify --local origin auth basic
dvc remote modify --local origin user "$DAGSHUB_USER"
dvc remote modify --local origin password "$DAGSHUB_TOKEN"

echo "Downloading model artifacts from DagsHub..."
dvc pull

echo "Starting Weather Forecasting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-10000}"