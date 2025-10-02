# Simple image to run the load tester with optional metrics endpoint
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install curl for optional script fetch and basic build deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY stress-test/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the load testing code
COPY stress-test/load_test.py ./load_test.py

# Default envs (can be overridden in Deployment)
ENV TARGET=both \
    DURATION=1800 \
    HPA_URL=http://product-app-hpa-service.default.svc.cluster.local \
    COMBINED_URL=http://product-app-combined-service.default.svc.cluster.local \
    PREDICTIVE_URL=http://predictive-scaler.default.svc.cluster.local:5000 \
    TEST_PREDICTIVE=false \
    MAX_CONCURRENCY=64 \
    METRICS_PORT=9105 \
    NORMAL_MIN=2 \
    NORMAL_MAX=5 \
    PEAK_MIN=100 \
    PEAK_MAX=200 \
    SEASON_DURATION=300 \
    PEAK_DURATION=60 \
    PEAK_OFFSET=60 \
    VOLATILITY=0.3 \
    TIMEOUT=60 \
    LOAD_TEST_SOURCE_URL=

# Entrypoint script builds command and optionally fetches latest script
COPY stress-test/entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 9105

CMD ["/app/entrypoint.sh"]
