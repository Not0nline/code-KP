# Simple image to run the load tester with optional metrics endpoint
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install curl and bash for optional script fetch and shell entrypoint
RUN apt-get update && apt-get install -y --no-install-recommends curl bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt flask

# Copy the load testing code
COPY load_test.py ./load_test.py

# Entrypoint script builds command and optionally fetches latest script
COPY entrypoint.sh ./entrypoint.sh
# Normalize Windows line endings and make executable
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 9105

CMD ["/app/entrypoint.sh"]
