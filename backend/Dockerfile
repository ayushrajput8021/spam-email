# Use multi-stage build for minimal final image size
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Environment variables to reduce size and improve build speed
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with --no-deps to avoid extra dependencies
RUN pip install --no-cache-dir --user --no-deps -r requirements.txt && \
    pip install --no-cache-dir --user numpy onnxruntime

# Final stage with minimal footprint
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

# Create a non-root user to run the app
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Copy only the necessary files from the builder stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY app/ /app/app/
COPY saved-model/ /app/saved-model/

# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["python", "app/main.py"]
