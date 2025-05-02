# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
# It's often better to copy requirements first to leverage Docker cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask prometheus-flask-exporter prometheus-client Werkzeug gunicorn

# If you don't have a requirements.txt, that's covered by the pip install above

# Copy the rest of the application code
COPY . .

# wget is already installed in a previous step, so this is not needed

# Expose the port Gunicorn will run on
EXPOSE 5000

# Command to run the application using Gunicorn
# -w 4: Use 4 worker processes (adjust based on pod CPU resources, 2-4 is common per core)
# -b 0.0.0.0:5000: Bind to all interfaces on port 5000
# --timeout 60: Kill worker if request takes longer than 60 seconds (adjust as needed)
# app:app : Look for the Flask app instance named 'app' in the file 'app.py'
CMD ["gunicorn", \
     "--workers", "${GUNICORN_WORKERS:-4}", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "10", \
     "app:app"]