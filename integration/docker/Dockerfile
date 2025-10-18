# Production Docker Image for Axolotl USF Integration Layer
# Based on Axolotl's Dockerfile-base pattern with CUDA 12.4
#
# Usage:
#   docker build -f integration/docker/Dockerfile -t axolotl-usf:latest .
#   docker run --gpus all -v $(pwd)/job_config.yaml:/workspace/job_config.yaml axolotl-usf:latest

# Use Axolotl's exact proven CUDA version from docker/Dockerfile-base
ARG CUDA_VERSION="11.8.0"
ARG CUDNN_VERSION="8"
ARG UBUNTU_VERSION="22.04"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

LABEL maintainer="Axolotl USF Team"
LABEL description="Production-ready automated fine-tuning with Axolotl"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/miniconda3/bin:${PATH}"

# Build arguments (matching Axolotl's docker/Dockerfile-base exactly)
ARG PYTHON_VERSION="3.11"
ARG PYTORCH_VERSION="2.1.2"
ARG CUDA="118"
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"

ENV PYTHON_VERSION=$PYTHON_VERSION
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

# Install system dependencies (following Axolotl's docker/Dockerfile-base pattern)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget git build-essential ninja-build git-lfs libaio-dev pkg-config \
        ibverbs-providers ibverbs-utils infiniband-diags \
        librdmacm-dev librdmacm1 rdmacm-utils \
        vim curl nano libnccl2 libnccl-dev rsync \
    && rm -rf /var/cache/apt/archives \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (exactly like Axolotl does)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

# Activate conda environment
ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

WORKDIR /workspace

# Install PyTorch with CUDA 12.4 support (following Axolotl's pattern)
RUN python3 -m pip install --upgrade pip && \
    pip3 install -U packaging setuptools wheel && \
    python3 -m pip install --no-cache-dir -U \
        torch==${PYTORCH_VERSION} \
        torchvision \
        torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA}

# Install Flash Attention (no-build-isolation like Axolotl)
RUN pip3 install --no-build-isolation flash-attn==2.6.3

# Install git-lfs and awscli (like Axolotl base)
RUN git lfs install --skip-repo && \
    pip3 install awscli && \
    pip3 install -U --no-cache-dir pydantic

# Copy Axolotl source code from repository
COPY . /workspace/axolotl-usf

WORKDIR /workspace/axolotl-usf

# Install Axolotl from source (following docker/Dockerfile pattern)
RUN pip3 install --no-build-isolation -e .[deepspeed,flash-attn,optimizers] && \
    python scripts/unsloth_install.py | sh && \
    python scripts/cutcrossentropy_install.py | sh && \
    pip3 cache purge

# Install integration layer dependencies
RUN pip3 install --no-cache-dir \
    boto3 \
    azure-storage-blob \
    google-cloud-storage \
    pymongo \
    wandb \
    mlflow \
    tensorboard \
    huggingface_hub[cli] \
    fastapi \
    uvicorn[standard] \
    pyyaml \
    python-dotenv

# Create workspace directories
RUN mkdir -p \
    /workspace/data \
    /workspace/output \
    /workspace/checkpoints \
    /workspace/logs

# Copy Docker-specific scripts
COPY integration/docker/entrypoint.sh /entrypoint.sh
COPY integration/docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /entrypoint.sh /healthcheck.sh

# Set working directory
WORKDIR /workspace

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/healthcheck.sh"]

# Expose ports
# 29500: PyTorch distributed training
# 8000: Optional API/monitoring endpoint
EXPOSE 29500
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run training wrapper
CMD ["python3", "-u", "/workspace/axolotl-usf/integration/train_wrapper.py", "--job-config", "/workspace/job_config.yaml"]
