# Use Python 3.10.6 slim image as base
FROM python:3.10.6-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip and install build tools for dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose the port your Flask app runs on
EXPOSE 10000

# Command to run the app
CMD ["python", "app.py"]
