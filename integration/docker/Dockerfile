# Production Docker Image for Axolotl USF Integration Layer
# Based on Axolotl's Dockerfile-base-next pattern (Python 3.11 + CUDA 12.8.1)
#
# Usage:
#   docker build -f integration/docker/Dockerfile -t axolotl-usf:latest .
#   docker run --gpus all -v $(pwd)/job_config.yaml:/workspace/job_config.yaml axolotl-usf:latest

# Use CUDA 11.8.0 (verified to exist and work from Axolotl's docker/Dockerfile-base)
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

# Build arguments (Python 3.11 with CUDA 11.8 - using stable PyTorch)
ARG PYTHON_VERSION="3.11"
ARG PYTORCH_VERSION="2.3.0"
ARG CUDA="118"
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"

ENV PYTHON_VERSION=$PYTHON_VERSION
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

# Install system dependencies and Miniconda (EXACTLY like Axolotl's Dockerfile-base-next)
RUN apt-get update \
    && apt-get install -y wget git build-essential ninja-build git-lfs libaio-dev pkg-config && rm -rf /var/lib/apt/lists/* \
    && wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

# Activate conda environment
ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

WORKDIR /workspace

# Install PyTorch and dependencies (Python 3.11 + CUDA 11.8)
RUN python3 -m pip install --upgrade pip && pip3 install packaging setuptools wheel && \
    python3 -m pip install --no-cache-dir -U torch==${PYTORCH_VERSION}+cu${CUDA} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu$CUDA && \
    python3 -m pip install --no-cache-dir "causal_conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@main" && \
    python3 -m pip install --no-cache-dir "mamba_ssm @ git+https://github.com/state-spaces/mamba.git@main"

# Install git-lfs, awscli, and pydantic
RUN git lfs install --skip-repo && \
    pip3 install awscli && \
    pip3 install -U --no-cache-dir pydantic

# Install Flash Attention for CUDA 12.1
RUN pip3 install --no-build-isolation flash-attn==2.6.3

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
