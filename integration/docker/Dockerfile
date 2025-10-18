# Production Docker Image for Axolotl Integration Layer
# Builds on top of official Axolotl image with minimal additions
#
# Usage:
#   docker build -f integration/docker/Dockerfile -t axolotl-usf:latest .
#   docker run --gpus all -v $(pwd)/job_config.yaml:/workspace/job_config.yaml axolotl-usf:latest

ARG AXOLOTL_VERSION=main
FROM axolotlai/axolotl:${AXOLOTL_VERSION}

LABEL maintainer="Axolotl USF Team"
LABEL description="Production-ready automated fine-tuning with Axolotl"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install integration layer dependencies
# Keep this layer minimal to optimize caching
RUN pip install --no-cache-dir \
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

# Copy integration layer code
COPY integration/ /workspace/integration/

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
