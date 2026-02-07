# TP-AI Docker Setup

This directory contains a complete Docker environment for your TensorFlow/Keras project with NVIDIA GPU support.

## 🚀 Quick Start

### 1. Start the Container
```bash
cd docker
./run.sh
```

### 2. Test the Environment
```bash
docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && python test_environment.py'
```

### 3. Run Your Training Script
```bash
docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && python models/mlp.py'
```

## 📁 Files Created

- `docker/docker-compose.yml` - Main Docker Compose configuration
- `docker/run.sh` - Helper script to start the container
- `docker/README.md` - Detailed documentation
- `docker/test_environment.py` - Environment test script
- `DOCKER_SETUP.md` - This summary file

## 🎯 Key Features

✅ **Full project access** - Your entire project directory is mounted inside the container  
✅ **NVIDIA GPU support** - CUDA acceleration for faster training  
✅ **TensorFlow/Keras ready** - All dependencies pre-installed  
✅ **Jupyter support** - Optional notebook server  
✅ **Easy commands** - Simple docker-compose commands  

## 🖥️ Container Details

**Container Name:** `tp-ai-container`  
**Working Directory:** `/workspace` (your project directory)  
**TensorFlow Environment:** `/home/docker/venv_tf/`  
**Jupyter Port:** `8888` (optional)

## 🔧 Available Commands

```bash
# Start container
docker-compose up -d

# Stop container  
docker-compose down

# Access container shell
docker-compose exec tp-ai-container bash

# Run your training script
docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && python models/mlp.py'

# Start Jupyter notebook
docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root'

# View logs
docker-compose logs tp-ai-container
```

## 📊 Inside the Container

Once inside the container shell:
```bash
# Activate TensorFlow environment
source /home/docker/venv_tf/bin/activate

# Your project files are in /workspace
ls /workspace

# Run your scripts
python models/mlp.py

# Check GPU status
nvidia-smi
```

## ✅ Prerequisites

- Docker installed
- Docker Compose installed  
- NVIDIA Docker runtime (`nvidia-docker2`)
- NVIDIA GPU with CUDA support

## 🐛 Troubleshooting

### GPU Not Working
```bash
# Check NVIDIA Docker
docker info | grep -i nvidia

# Test NVIDIA container
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

### Permission Issues
```bash
chmod +x docker/run.sh
```

### Container Won't Start
```bash
sudo systemctl status docker
sudo systemctl status nvidia-docker
```

## 🎉 You're Ready!

Your Docker environment is now set up with:
- Full access to all your project files
- NVIDIA GPU acceleration
- TensorFlow/Keras with all dependencies
- Easy-to-use commands

Start training your neural networks with GPU acceleration! 🚀