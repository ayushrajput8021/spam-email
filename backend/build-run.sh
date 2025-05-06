#!/bin/bash

# Exit on error
set -e

# Build the optimized Docker image
echo "Building optimized spam-email-detector image..."
docker build -t spam-email-detector:slim -f Dockerfile .

# Show image size
echo "Image size:"
docker images spam-email-detector:slim --format "{{.Size}}"

# Run the container
echo "Running container..."
docker run -p 8000:8000 --name spam-detector -d spam-email-detector:slim

echo "API is running at http://localhost:8000"
echo "To check logs: docker logs spam-detector"
echo "To stop container: docker stop spam-detector"
