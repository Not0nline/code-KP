FROM python:3.9-slim

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir requests kubernetes prometheus_client

# Copy the controller script
COPY controller.py .

# Set environment variables with defaults (will be overridden by deployment)
ENV PREDICTIVE_SCALER_SERVICE="http://predictive-scaler:5000/predict" \
    TARGET_DEPLOYMENT="product-app" \
    TARGET_NAMESPACE="default" \
    POLL_INTERVAL="60"

# Run the controller
CMD ["python", "-u", "/app/controller.py"]