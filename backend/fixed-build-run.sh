#!/bin/bash

# Exit on error
set -e

# Build the Docker image using the fixed Dockerfile
echo "Building spam-email-detector image with fixed Dockerfile..."
docker build -t spam-email-detector:fixed -f Dockerfile.fixed .

# Show image size
echo "Image size:"
docker images spam-email-detector:fixed --format "{{.Size}}"

# Stop existing container if running
echo "Stopping any existing container..."
docker stop spam-detector 2>/dev/null || true
docker rm spam-detector 2>/dev/null || true

# Run the container
echo "Running container..."
docker run -p 8000:8000 --name spam-detector -d spam-email-detector:fixed

echo "API is running at http://localhost:8000"
echo "To check logs: docker logs spam-detector"
echo "To test endpoint: curl -X POST -H 'Content-Type: application/json' -d '{\"email_content\":\"Free money now!\"}' http://localhost:8000/predict"
echo "To stop container: docker stop spam-detector"
