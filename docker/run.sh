f c#!/bin/bash

# TP-AI Docker Container Runner
# This script helps you run the TensorFlow/Keras environment

set -e

echo "=== TP-AI Docker Environment ==="
echo "Starting container with NVIDIA GPU support..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose not found. Please install docker-compose."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "Warning: NVIDIA Docker runtime not detected."
    echo "GPU acceleration may not work properly."
    echo "Make sure you have nvidia-docker2 installed."
fi

# Start the container
echo "Starting container..."
docker-compose up -d

# Wait a moment for container to be ready
sleep 3

echo ""
echo "Container started successfully!"
echo ""
echo "Available commands:"
echo "  docker-compose exec tp-ai-container bash    # Start interactive shell"
echo "  docker-compose logs tp-ai-container         # View container logs"
echo "  docker-compose down                         # Stop container"
echo ""
echo "To run your training script:"
echo "  docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && python models/mlp.py'"
echo ""
echo "To start Jupyter notebook (optional):"
echo "  docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root'"
echo "  Then open http://localhost:8888 in your browser"
echo ""
echo "Container logs:"
docker-compose logs tp-ai-container