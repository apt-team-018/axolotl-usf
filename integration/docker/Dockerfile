# Production Docker Image for Axolotl Integration Layer
# Builds Axolotl from source and adds integration layer
#
# Usage:
#   docker build -f integration/docker/Dockerfile -t axolotl-usf:latest .
#   docker run --gpus all -v $(pwd)/job_config.yaml:/workspace/job_config.yaml axolotl-usf:latest

# Use NVIDIA CUDA base image (verified to exist)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="Axolotl USF Team"
LABEL description="Production-ready automated fine-tuning with Axolotl"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install system dependencies including Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils \
    git wget curl vim nano \
    build-essential ninja-build \
    libaio-dev pkg-config \
    infiniband-diags ibverbs-providers ibverbs-utils \
    librdmacm-dev librdmacm1 rdmacm-utils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 11.8 base + newer drivers)
RUN pip3 install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention
RUN pip3 install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# Set working directory
WORKDIR /workspace

# Copy Axolotl source code from current repository
COPY . /workspace/axolotl-usf

# Install Axolotl from source
WORKDIR /workspace/axolotl-usf
RUN pip3 install --no-cache-dir -e .[deepspeed,flash-attn,optimizers]

# Install integration layer dependencies
RUN pip3 install --no-cache-dir \
    # Cloud storage clients
    boto3==1.34.52 \
    azure-storage-blob==12.19.0 \
    google-cloud-storage==2.14.0 \
    # Monitoring tools
    pymongo==4.6.1 \
    wandb==0.16.3 \
    mlflow==2.10.2 \
    tensorboard==2.16.2 \
    # API dependencies (for future API endpoints)
    fastapi==0.109.2 \
    uvicorn[standard]==0.27.1 \
    # HuggingFace Hub with CLI
    huggingface_hub[cli]==0.20.3 \
    # Additional utilities
    pyyaml==6.0.1 \
    python-dotenv==1.0.1

# Create required directories
RUN mkdir -p \
    /workspace/data \
    /workspace/output \
    /workspace/checkpoints \
    /workspace/logs

# Integration layer is already copied as part of the repo
# No need to copy again

# Copy Docker-specific scripts
COPY integration/docker/entrypoint.sh /entrypoint.sh
COPY integration/docker/healthcheck.sh /healthcheck.sh

# Make scripts executable
RUN chmod +x /entrypoint.sh /healthcheck.sh

# Set working directory
WORKDIR /workspace

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/healthcheck.sh"]

# Expose common ports
# 29500: PyTorch distributed training
# 8000: Optional API/monitoring endpoint
EXPOSE 29500
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run training wrapper
CMD ["python3", "-u", "/workspace/integration/train_wrapper.py", "--job-config", "/workspace/job_config.yaml"]
