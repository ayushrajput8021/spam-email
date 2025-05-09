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

# Install Python dependencies and verify uvicorn installation
RUN pip install --no-cache-dir --user -r requirements.txt && \
    pip list | grep uvicorn && \
    ls -la /root/.local/bin/uvicorn

# Final stage with minimal footprint
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH

# Create a non-root user to run the app
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Copy installed packages from builder (ensure correct permissions)
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser saved-model/ /app/saved-model/

# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Memory optimization for Python
ENV PYTHONMALLOC=malloc \
    PYTHONVM_STAT_INTERVAL=7200

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
