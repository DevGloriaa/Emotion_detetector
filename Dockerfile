# Use Python 3.10.6 slim image
FROM python:3.10.6-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies, Python build tools, and pin pip
RUN apt-get update && apt-get install -y \
  build-essential \
  python3.10-dev \
  python3.10-distutils \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  && pip install "pip<24.1" setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Expose your app port
EXPOSE 10000

# Run the app
CMD ["python", "app.py"]
