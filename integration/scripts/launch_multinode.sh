#!/bin/bash
# Generic multi-node training launcher
# Works on ANY platform with networked GPUs

set -e

# ============================================================================
# Configuration
# ============================================================================

# Required environment variables:
# - NUM_NODES: Number of nodes
# - GPUS_PER_NODE: GPUs per node
# - MASTER_ADDR: Master node address
# - MASTER_PORT: Master node port
# - JOB_CONFIG: Path to job config YAML
# - NODE_RANK: Rank of this node (0 for master, 1,2,3... for workers)

# ============================================================================
# Validation
# ============================================================================

if [ -z "$NUM_NODES" ] || [ -z "$GPUS_PER_NODE" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
    echo "Error: Required environment variables not set"
    echo "Required: NUM_NODES, GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT"
    exit 1
fi

if [ -z "$JOB_CONFIG" ]; then
    echo "Error: JOB_CONFIG environment variable not set"
    exit 1
fi

# Set NODE_RANK if not set (default to 0 for backward compatibility)
NODE_RANK=${NODE_RANK:-0}

# ============================================================================
# Environment Setup
# ============================================================================

# Export distributed training variables
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
export RANK=$((NODE_RANK * GPUS_PER_NODE))

echo "=========================================="
echo "Launching Multi-Node Training"
echo "=========================================="
echo "Configuration:"
echo "  Nodes: $NUM_NODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  World size: $WORLD_SIZE"
echo "  Node rank: $NODE_RANK"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Job config: $JOB_CONFIG"
echo "=========================================="

# ============================================================================
# Launch Training
# ============================================================================

# Use torchrun for robust distributed launching
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf="join_timeout=1800" \
    /workspace/integration/train_wrapper.py \
    --job-config=$JOB_CONFIG

echo "=========================================="
echo "Training completed on node $NODE_RANK"
echo "=========================================="
