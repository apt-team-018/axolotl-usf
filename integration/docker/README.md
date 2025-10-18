# Axolotl USF Docker Documentation

Complete Docker setup for running Axolotl fine-tuning with the USF integration layer.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Single GPU Testing](#single-gpu-testing)
  - [Multi-Node Training](#multi-node-training)
- [Architecture](#architecture)
- [Building Images](#building-images)
- [Running Locally](#running-locally)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Docker setup provides production-ready containerization for Axolotl training with:

- **Lightweight Design**: Builds on top of official `axolotlai/axolotl` images (~500MB addition)
- **Production Features**: Health checks, monitoring, distributed training support
- **Flexible Deployment**: Local, Docker Compose, Kubernetes, AWS ECS
- **Complete Integration**: Storage backends, monitoring tools, lifecycle management

### Architecture

```
axolotlai/axolotl:main (Official - ~10GB)
    ↓
axolotl-usf:latest (Integration Layer - ~10.5GB)
    ├── Integration code
    ├── Storage clients (S3, Azure, GCS)
    ├── Monitoring tools (W&B, MLflow, MongoDB)
    └── Enhanced entrypoint & health checks
```

## 🚀 Quick Start

### Single GPU Testing

**Perfect for**: RunPod, Vast.ai, Azure, AWS, or local GPU testing

Complete guide with platform-specific instructions: **[Single GPU QuickStart →](examples/SINGLE_GPU_QUICKSTART.md)**

```bash
# 3-command quickstart (Qwen2.5-0.5B + LoRA + 100 samples)
bash integration/docker/build.sh

docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:latest

ls -lh output/
```

### Multi-Node Training

**Perfect for**: 4+ node distributed training

Complete guide with 4-node example: **[Multi-Node Setup →](examples/MULTI_NODE_EXAMPLE.md)**

## 🎯 Detailed Quick Start

### Prerequisites

- Docker 20.10+
- NVIDIA Docker runtime (for GPU support)
- Docker Compose 2.0+ (optional)

### 1. Build the Image

```bash
# From axolotl-usf root directory
cd /path/to/axolotl-usf

# Build with default settings
bash integration/docker/build.sh

# Or build with specific Axolotl version
bash integration/docker/build.sh --axolotl-version nightly --tag nightly
```

### 2. Prepare Configuration

```bash
# Create job configuration
cp integration/examples/quickstart_local.yaml job_config.yaml

# Edit configuration for your needs
vim job_config.yaml
```

### 3. Run Training

```bash
# Simple Docker run
docker run --gpus all \
  -v $(pwd)/job_config.yaml:/workspace/job_config.yaml \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:latest

# Or use Docker Compose
docker-compose -f integration/docker/docker-compose.yml up
```

## 🏗️ Building Images

### Basic Build

```bash
bash integration/docker/build.sh
```

### Advanced Build Options

```bash
# Specify Axolotl version
bash integration/docker/build.sh --axolotl-version main

# Build with custom tag
bash integration/docker/build.sh --tag v1.0.0

# Build and push to registry
bash integration/docker/build.sh \
  --registry ghcr.io/myorg \
  --tag latest \
  --push

# No-cache build (force rebuild)
bash integration/docker/build.sh --no-cache
```

### Manual Build

```bash
# Build from Dockerfile directly
docker build \
  --build-arg AXOLOTL_VERSION=main \
  -f integration/docker/Dockerfile \
  -t axolotl-usf:latest \
  .
```

## 💻 Running Locally

### Option 1: Direct Docker Run

```bash
docker run --gpus all \
  --name axolotl-training \
  -v $(pwd)/job_config.yaml:/workspace/job_config.yaml:ro \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e LOG_LEVEL=INFO \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  axolotl-usf:latest
```

### Option 2: Docker Compose (Recommended)

```bash
# Single-node training
docker-compose -f integration/docker/docker-compose.yml up

# With monitoring stack (MongoDB + MLflow)
docker-compose -f integration/docker/docker-compose.yml --profile monitoring up

# Background mode
docker-compose -f integration/docker/docker-compose.yml up -d

# View logs
docker-compose -f integration/docker/docker-compose.yml logs -f

# Stop training
docker-compose -f integration/docker/docker-compose.yml down
```

### Option 3: Multi-Node Distributed Training

```bash
# Terminal 1 - Master node (rank 0)
WORLD_SIZE=2 GPUS_PER_NODE=2 MASTER_ADDR=localhost \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master

# Terminal 2 - Worker node (rank 1)
WORLD_SIZE=2 GPUS_PER_NODE=2 MASTER_ADDR=localhost NODE_RANK=1 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

## ☁️ Deployment Options

### Kubernetes

Deploy to Kubernetes cluster with GPU support:

```bash
# Apply deployment
kubectl apply -f integration/docker/examples/kubernetes-deployment.yaml

# Monitor training
kubectl logs -f deployment/axolotl-usf-training -n axolotl-usf

# Check status
kubectl get pods -n axolotl-usf
```

### AWS ECS

1. **Push image to ECR**:
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag axolotl-usf:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/axolotl-usf:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/axolotl-usf:latest
```

2. **Register task definition**:
```bash
# Update account ID and region in task definition
vim integration/docker/examples/aws-ecs-task-definition.json

# Register task
aws ecs register-task-definition \
  --cli-input-json file://integration/docker/examples/aws-ecs-task-definition.json
```

3. **Run task**:
```bash
aws ecs run-task \
  --cluster your-gpu-cluster \
  --task-definition axolotl-usf-training \
  --count 1 \
  --launch-type EC2
```

### RunPod / Vast.ai

Use the Docker image directly on GPU cloud platforms:

```bash
# RunPod: Use in template configuration
# Image: axolotl-usf:latest
# Docker Command: --job-config /workspace/job_config.yaml

# Vast.ai: Launch instance with custom image
# Instance Image: your-registry/axolotl-usf:latest
# On-start script: Upload job_config.yaml to /workspace/
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from the example:

```bash
cp integration/docker/examples/.env.example .env
vim .env
```

Key variables:

```bash
# Training
WORLD_SIZE=1
GPUS_PER_NODE=1

# Monitoring
WANDB_API_KEY=your_key_here
MLFLOW_TRACKING_URI=http://localhost:5000

# Storage
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Logging
LOG_LEVEL=INFO
JSON_LOGS=true
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./job_config.yaml` | `/workspace/job_config.yaml` | Job configuration (read-only) |
| `./data` | `/workspace/data` | Dataset storage |
| `./output` | `/workspace/output` | Trained model output |
| `./checkpoints` | `/workspace/checkpoints` | Training checkpoints |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace model cache |

### Health Checks

The container includes automatic health checks:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' axolotl-training

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' axolotl-training
```

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check Docker configuration
cat /etc/docker/daemon.json
# Should contain: "default-runtime": "nvidia"
```

#### Out of Memory

```bash
# Increase shared memory in docker-compose.yml
shm_size: '32gb'

# Or in docker run
docker run --shm-size=32g ...

# Reduce batch size in job_config.yaml
hyperparameters:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
```

#### Slow Build Times

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker build ...

# Use cache from registry
docker build --cache-from axolotl-usf:latest ...
```

#### Network Issues (Multi-Node)

```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check network connectivity
docker exec axolotl-training ping master-node-ip

# Verify NCCL can use network interface
export NCCL_SOCKET_IFNAME=eth0
```

### Debug Mode

Run container with interactive shell:

```bash
docker run --gpus all -it \
  -v $(pwd)/job_config.yaml:/workspace/job_config.yaml \
  axolotl-usf:latest \
  /bin/bash

# Inside container
python3 /workspace/integration/train_wrapper.py --job-config /workspace/job_config.yaml --validate-only
```

### View Logs

```bash
# Docker logs
docker logs -f axolotl-training

# Docker Compose logs
docker-compose -f integration/docker/docker-compose.yml logs -f

# Export logs
docker logs axolotl-training > training.log 2>&1
```

## 📊 Monitoring

### Container Metrics

```bash
# Resource usage
docker stats axolotl-training

# GPU utilization
docker exec axolotl-training nvidia-smi
```

### Training Metrics

Access monitoring dashboards:

- **Weights & Biases**: https://wandb.ai
- **MLflow**: http://localhost:5000 (if using monitoring profile)
- **TensorBoard**: `tensorboard --logdir /workspace/logs`

## 🔐 Security Best Practices

1. **Never commit secrets**:
   - Use `.env` files (git-ignored)
   - Use secrets managers (AWS Secrets Manager, HashiCorp Vault)
   - Use IAM roles instead of credentials when possible

2. **Use read-only mounts** for configs:
   ```yaml
   volumes:
     - ./job_config.yaml:/workspace/job_config.yaml:ro
   ```

3. **Run as non-root** (future enhancement):
   ```dockerfile
   USER 1000:1000
   ```

4. **Scan images for vulnerabilities**:
   ```bash
   docker scan axolotl-usf:latest
   ```

## 📚 Additional Resources

- [Axolotl Documentation](https://github.com/axolotl-ai-cloud/axolotl)
- [Integration Layer Docs](../docs/README.md)
- [Job Configuration Guide](../docs/JOB_CONFIG.md)
- [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)

## 🆘 Support

For issues specific to Docker deployment:
1. Check this troubleshooting guide
2. Review container logs
3. Test with validation-only mode
4. Open an issue with full error logs and configuration

---

**Next Steps**: Review [deployment examples](examples/) for your cloud platform.
