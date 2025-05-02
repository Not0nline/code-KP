FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Create directories for data persistence
RUN mkdir -p /app/data

EXPOSE 5000 

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_FILE=/app/data/traffic_data.csv
ENV MODEL_FILE=/app/data/gru_model
ENV CONFIG_FILE=/app/data/config.json

# Use Gunicorn to bind to port 5000 explicitly
CMD ["sh", "-c", "if [ \"$ENVIRONMENT\" = \"dev\" ]; then python app.py; else gunicorn --bind 0.0.0.0:5000 app:app; fi"]