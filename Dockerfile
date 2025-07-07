FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ backend/
COPY __init__.py .
COPY backend/__init__.py backend/

# Use PORT environment variable for Railway
CMD uvicorn backend.app:app --host 0.0.0.0 --port $PORT