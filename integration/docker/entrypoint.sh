#!/bin/bash
# Enhanced entrypoint for Axolotl USF Docker container
# Handles initialization, validation, and graceful execution

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Axolotl USF - Automated Fine-Tuning${NC}"
echo -e "${BLUE}Version: 1.0.0${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Display system information
echo -e "${GREEN}System Information:${NC}"
echo "  Hostname: $(hostname)"
echo "  User: $(whoami)"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "  Axolotl: $(python3 -c 'import axolotl; print(axolotl.__version__)' 2>/dev/null || echo 'N/A')"
echo ""

# Display distributed training environment
echo -e "${GREEN}Distributed Training Environment:${NC}"
echo "  WORLD_SIZE: ${WORLD_SIZE:-1}"
echo "  RANK: ${RANK:-0}"
echo "  LOCAL_RANK: ${LOCAL_RANK:-0}"
echo "  NODE_RANK: ${NODE_RANK:-0}"
echo "  MASTER_ADDR: ${MASTER_ADDR:-localhost}"
echo "  MASTER_PORT: ${MASTER_PORT:-29500}"
echo ""

# Display GPU information
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU Information:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader | while read line; do
        echo "  GPU: $line"
    done
    echo ""

    # Display CUDA environment
    echo -e "${GREEN}CUDA Environment:${NC}"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
    echo "  NCCL_DEBUG: ${NCCL_DEBUG:-WARN}"
    echo ""
elif command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}AMD GPU Information:${NC}"
    rocm-smi --showproductname
    echo ""
else
    echo -e "${YELLOW}⚠️  No GPU detected (CPU-only mode)${NC}"
    echo ""
fi

# Check for required directories
echo -e "${GREEN}Checking workspace directories:${NC}"
for dir in /workspace/data /workspace/output /workspace/checkpoints /workspace/logs; do
    if [ -d "$dir" ]; then
        echo -e "  ${GREEN}✓${NC} $dir"
    else
        echo -e "  ${YELLOW}⚠${NC}  Creating $dir"
        mkdir -p "$dir"
    fi
done
echo ""

# Display storage/monitoring configuration (if set)
if [ -n "$WANDB_API_KEY" ]; then
    echo -e "${GREEN}✓${NC} Weights & Biases configured"
fi
if [ -n "$MLFLOW_TRACKING_URI" ]; then
    echo -e "${GREEN}✓${NC} MLflow configured: $MLFLOW_TRACKING_URI"
fi
if [ -n "$MONGODB_URI" ]; then
    echo -e "${GREEN}✓${NC} MongoDB configured"
fi
if [ -n "$AWS_REGION" ] || [ -n "$AWS_DEFAULT_REGION" ]; then
    echo -e "${GREEN}✓${NC} AWS configured: ${AWS_REGION:-$AWS_DEFAULT_REGION}"
fi
echo ""

# Validate job config if default path is used
if [ $# -eq 0 ] && [ -f "/workspace/job_config.yaml" ]; then
    echo -e "${GREEN}Validating job configuration:${NC}"
    if python3 /workspace/integration/train_wrapper.py --job-config /workspace/job_config.yaml --validate-only 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Job configuration is valid"
    else
        echo -e "${RED}✗${NC} Job configuration validation failed"
        echo -e "${YELLOW}Proceeding anyway (validation errors may cause training to fail)${NC}"
    fi
    echo ""
fi

# Handle signals for graceful shutdown
trap 'echo -e "\n${YELLOW}Received SIGTERM, shutting down gracefully...${NC}"; exit 143' TERM
trap 'echo -e "\n${YELLOW}Received SIGINT, shutting down gracefully...${NC}"; exit 130' INT

# Execute the provided command or default command
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Starting Training${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

if [ $# -eq 0 ]; then
    # Default: Run training wrapper
    echo -e "${GREEN}Executing:${NC} python3 /workspace/integration/train_wrapper.py --job-config /workspace/job_config.yaml"
    exec python3 -u /workspace/integration/train_wrapper.py --job-config /workspace/job_config.yaml
else
    # Custom command provided
    echo -e "${GREEN}Executing:${NC} $@"
    exec "$@"
fi
