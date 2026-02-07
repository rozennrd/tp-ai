# TP-AI Docker Environment

This directory contains the Docker configuration for running your TensorFlow/Keras project with NVIDIA GPU support.

## Quick Start

### Prerequisites

1. **Docker** installed
2. **Docker Compose** installed
3. **NVIDIA Docker** runtime (`nvidia-docker2`)
4. **NVIDIA GPU** with CUDA support

### Installation

1. Install NVIDIA Docker runtime:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. Verify NVIDIA Docker is working:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
   ```

### Usage

#### Option 1: Use the helper script (Recommended)
```bash
cd docker
./run.sh
```

#### Option 2: Use docker-compose directly
```bash
cd docker
docker-compose up -d
```

### Running Your Project

Once the container is running, you have several options:

#### 1. Interactive Shell
```bash
docker-compose exec tp-ai-container bash
# Inside the container:
source /home/docker/venv_tf/bin/activate
python models/mlp.py
```

#### 2. Run Training Script Directly
```bash
docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && python models/mlp.py'
```

#### 3. Start Jupyter Notebook (Optional)
```bash
docker-compose exec tp-ai-container bash -c 'source /home/docker/venv_tf/bin/activate && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root'
```
Then open http://localhost:8888 in your browser.

### Container Features

- **Full project access**: Your entire project directory is mounted at `/workspace`
- **GPU acceleration**: NVIDIA GPU support with CUDA
- **TensorFlow environment**: Pre-installed TensorFlow with CUDA support
- **All dependencies**: NumPy, Matplotlib, PIL, SciPy, etc.
- **Jupyter support**: Optional Jupyter notebook server

### Available Commands

```bash
# Start container
docker-compose up -d

# Stop container
docker-compose down

# View logs
docker-compose logs tp-ai-container

# Access container shell
docker-compose exec tp-ai-container bash

# Run specific command
docker-compose exec tp-ai-container <command>
```

### File Structure

Inside the container:
- `/workspace/` - Your project directory (contains all your files)
- `/home/docker/venv_tf/` - TensorFlow virtual environment
- `/home/docker/venv_torch/` - PyTorch virtual environment (optional)

### Troubleshooting

#### GPU Not Detected
```bash
# Check if NVIDIA Docker runtime is available
docker info | grep -i nvidia

# Test NVIDIA container
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

#### Permission Issues
```bash
# Make sure the run script is executable
chmod +x docker/run.sh
```

#### Container Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check NVIDIA Docker
sudo systemctl status nvidia-docker
```

## Customization

### Modify Dependencies
Edit the `Dockerfile` in this directory to add/remove packages.

### Change Working Directory
Modify the `working_dir` in `docker-compose.yml`.

### Add More Volumes
Add additional volume mounts in the `volumes` section of `docker-compose.yml`.

## Performance Notes

- The container uses your NVIDIA GPU for CUDA acceleration
- TensorFlow is configured to use GPU by default
- Monitor GPU usage with `nvidia-smi` inside the container
- The environment variables `TF_ENABLE_ONEDNN_OPTS=0` and `TF_CPP_MIN_LOG_LEVEL=2` are set to optimize performance and reduce logging