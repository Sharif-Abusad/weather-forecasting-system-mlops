FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

COPY start.sh .
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["sh", "start.sh"]