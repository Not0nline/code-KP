FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY baseline_datasets.py .
COPY model_variants.py .
COPY tree_models.py .
COPY statuscale_model.py .
COPY lstm_model.py .
COPY advanced_models.py .
COPY model-config.yaml .

# Create directories for data persistence
RUN mkdir -p /app/data

EXPOSE 5000 

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_FILE=/app/data/traffic_data.csv
ENV MODEL_FILE=/app/data/gru_model
ENV CONFIG_FILE=/app/data/config.json
ENV GUNICORN_WORKERS=2
ENV GUNICORN_THREADS=4
ENV GUNICORN_TIMEOUT=120
ENV GUNICORN_GRACEFUL=30
ENV GUNICORN_KEEPALIVE=5
ENV GUNICORN_CMD_ARGS="--worker-tmp-dir /dev/shm"

# Use Gunicorn to bind to port 5000 explicitly
CMD ["sh", "-c", "if [ \"$ENVIRONMENT\" = \"dev\" ]; then python app.py; else gunicorn --bind 0.0.0.0:5000 --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-120} --graceful-timeout ${GUNICORN_GRACEFUL:-30} --keep-alive ${GUNICORN_KEEPALIVE:-5} app:app; fi"]