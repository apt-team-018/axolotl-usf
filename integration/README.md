# Automated Fine-Tuning Integration Layer

Production-ready automated fine-tuning system built on top of Axolotl framework.

## 🎯 Features

- **Universal GPU Support**: NVIDIA (H200, B200, H100, A100) and AMD (MI300)
- **Massive Scalability**: 1-128+ nodes with 1-8 GPUs per node
- **Storage Flexibility**: S3, Azure Blob, GCS, Shared Filesystem, HuggingFace
- **Platform Agnostic**: Kubernetes, SLURM, Docker, RunPod, bare metal
- **Training Methods**: Full fine-tuning, LoRA, QLoRA
- **Comprehensive Monitoring**: W&B, MLflow, TensorBoard, MongoDB

## 📦 Installation

### Prerequisites

- Python 3.11+
- CUDA 12.4+ (for NVIDIA GPUs) or ROCm 6.0+ (for AMD GPUs)
- Docker (optional, but recommended)

### Quick Start with Docker

```bash
# Build NVIDIA image
cd integration/docker
docker build -f Dockerfile.nvidia -t axolotl-autotrain:nvidia-latest ../../

# Build AMD image
docker build -f Dockerfile.amd -t axolotl-autotrain:amd-latest ../../

# Run single-node training
docker run --rm --gpus all \
    -v $(pwd)/examples:/workspace/examples \
    -v $(pwd)/data:/workspace/data \
    axolotl-autotrain:nvidia-latest \
    python3 /workspace/integration/train_wrapper.py \
    --job-config /workspace/examples/job_config_qlora_single_node.yaml
```

## 🚀 Usage

### 1. Create Job Configuration

```yaml
# job_config.yaml
job_id: "my-training-job"
model: "meta-llama/Llama-3.1-8B"
training_type: "qlora"  # full, lora, or qlora

nodes: 1
gpus_per_node: 4

storage:
  type: "s3"
  credentials:
    aws_access_key_id: "YOUR_KEY"
    aws_secret_access_key: "YOUR_SECRET"
    region: "us-east-1"
  dataset_uri: "s3://bucket/data.jsonl"
  output_uri: "s3://bucket/models/output"

monitoring:
  wandb:
    project: "my-project"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "training"
    collection: "metrics"

hyperparameters:
  batch_size: 64
  learning_rate: 2e-4
  num_epochs: 3
```

### 2. Run Training

#### Single Node

```bash
python3 integration/train_wrapper.py --job-config job_config.yaml
```

#### Multi-Node with torchrun

```bash
# On each node, run:
export NUM_NODES=4
export GPUS_PER_NODE=8
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500
export NODE_RANK=<0,1,2,3>

torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    integration/train_wrapper.py \
    --job-config job_config.yaml
```

## 📁 Directory Structure

```
integration/
├── __init__.py
├── train_wrapper.py          # Main training entry point
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── manager.py            # Dynamic config generation
│   ├── validator.py          # Config validation
│   └── templates/            # Config templates
│       ├── base/             # Base training configs
│       │   ├── sft_full.yaml
│       │   ├── sft_lora.yaml
│       │   └── sft_qlora.yaml
│       └── hardware/         # Hardware configs
│           └── multi_node_fsdp.yaml
├── storage/                   # Storage abstraction
│   ├── __init__.py
│   ├── base.py               # Base interface
│   ├── manager.py            # Storage factory
│   ├── s3.py                 # AWS S3 backend
│   ├── azure.py              # Azure Blob backend
│   ├── gcs.py                # Google Cloud Storage
│   ├── filesystem.py         # Local/shared filesystem
│   └── huggingface.py        # HuggingFace Hub
├── monitoring/                # Monitoring & logging
│   ├── __init__.py
│   ├── mongodb_logger.py     # MongoDB logger
│   └── tracker.py            # Multi-backend tracker
├── distributed/               # Distributed training
│   ├── __init__.py
│   └── coordinator.py        # Platform-agnostic coordinator
├── docker/                    # Docker images
│   ├── Dockerfile.nvidia     # NVIDIA GPU image
│   ├── Dockerfile.amd        # AMD GPU image
│   └── entrypoint.sh         # Container entrypoint
└── examples/                  # Example configurations
    ├── job_config_qlora_single_node.yaml
    └── job_config_multi_node.yaml
```

## 🔧 Configuration Templates

### Training Types

1. **Full Fine-Tuning** (`training_type: full`)
   - Trains all model parameters
   - Highest memory usage
   - Best for small models or large GPU clusters

2. **LoRA** (`training_type: lora`)
   - Low-Rank Adaptation
   - Trains only adapter layers
   - 3-4x memory reduction

3. **QLoRA** (`training_type: qlora`)
   - Quantized LoRA (4-bit quantization)
   - Lowest memory usage
   - Enables fine-tuning of 70B+ models on consumer GPUs

### Storage Backends

```yaml
# S3
storage:
  type: "s3"
  credentials:
    aws_access_key_id: "..."
    aws_secret_access_key: "..."
    region: "us-east-1"
  dataset_uri: "s3://bucket/data.jsonl"
  output_uri: "s3://bucket/output"

# Azure Blob
storage:
  type: "azure"
  credentials:
    account_name: "..."
    account_key: "..."
  dataset_uri: "azure://container/data.jsonl"
  output_uri: "azure://container/output"

# HuggingFace Hub
storage:
  type: "hf"
  credentials:
    hf_token: "hf_..."
  dataset_uri: "hf://username/dataset"
  output_uri: "hf://username/model"

# Local/Shared Filesystem
storage:
  type: "filesystem"
  dataset_uri: "/data/dataset.jsonl"
  output_uri: "/models/output"
```

### Monitoring Backends

```yaml
monitoring:
  # Weights & Biases
  wandb:
    project: "my-project"
    entity: "my-team"
    tags: ["llama", "qlora"]

  # MLflow
  mlflow:
    tracking_uri: "https://mlflow.example.com"
    experiment_name: "llama-finetuning"

  # TensorBoard
  tensorboard:
    log_dir: "./runs"

  # MongoDB (custom metrics)
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "training"
    collection: "metrics"
```

## 🎯 Model Size Recommendations

| Model Size | Nodes | GPUs/Node | Training Type | Config |
|------------|-------|-----------|---------------|--------|
| 135M-1B | 1 | 1-2 | Full/LoRA | Single node |
| 3B-7B | 1 | 4-8 | LoRA/QLoRA | Single node |
| 13B-30B | 2-4 | 8 | QLoRA + FSDP | Multi-node |
| 70B | 4-8 | 8 | QLoRA + FSDP | Multi-node |
| 405B+ | 16-32 | 8 | QLoRA + FSDP + TP | Multi-node |

## 🐛 Troubleshooting

### Out of Memory

```yaml
# Reduce batch size
hyperparameters:
  micro_batch_size: 1
  gradient_accumulation_steps: 32

# Enable gradient checkpointing (always on by default)
# Switch to QLoRA if using LoRA
training_type: "qlora"

# Enable CPU offloading for very large models
hyperparameters:
  fsdp_offload: true
```

### Slow Training

```yaml
# Increase batch size to max GPU memory
hyperparameters:
  micro_batch_size: 4  # Increase until OOM

# Enable sample packing (enabled by default)
# Use smaller sequence length if appropriate
hyperparameters:
  sequence_length: 2048  # Instead of 4096
```

### Multi-Node Communication Issues

```bash
# Check network connectivity
ping <other-node-ip>

# Verify NCCL settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand

# Increase timeout
export NCCL_TIMEOUT=7200
```

## 📊 Monitoring

### Real-time Metrics

The system logs metrics to configured backends in real-time:

- **Training Loss**: Loss value at each step
- **Learning Rate**: Current learning rate
- **GPU Memory**: Memory usage per GPU
- **Throughput**: Tokens/second
- **Checkpoints**: Checkpoint save events

### MongoDB Query Examples

```javascript
// Get all metrics for a job
db.metrics.find({job_id: "my-job-id"})

// Get latest training step
db.metrics.find({
  job_id: "my-job-id",
  event_type: "training_step"
}).sort({timestamp: -1}).limit(1)

// Get average loss over last 100 steps
db.metrics.aggregate([
  {$match: {job_id: "my-job-id", event_type: "training_step"}},
  {$sort: {step: -1}},
  {$limit: 100},
  {$group: {_id: null, avg_loss: {$avg: "$loss"}}}
])
```

## 🔐 Security

- **Credentials**: Never commit credentials to git. Use environment variables.
- **Encryption**: All credentials should be encrypted at rest.
- **Network**: Use VPNs or private networks for multi-node setups.
- **Checkpoints**: Verify checksums when uploading/downloading.

## 📝 License

This integration layer follows the same license as Axolotl.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: See `/docs/AUTOMATED_FINETUNING_ARCHITECTURE.md`
- **Examples**: See `/integration/examples/`

## 🙏 Acknowledgments

Built on top of the excellent [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) framework.
