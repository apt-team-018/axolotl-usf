# Job Configuration Reference

Complete reference for job configuration files (`.yaml`).

## Table of Contents
- [Overview](#overview)
- [Required Fields](#required-fields)
- [Storage Configuration](#storage-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Hyperparameters](#hyperparameters)
- [Lifecycle Configuration](#lifecycle-configuration)
- [Secrets Management](#secrets-management)
- [Auto-Configuration](#auto-configuration)
- [Complete Example](#complete-example)

## Overview

The job configuration file defines all aspects of your training job:
- Model and training type
- Hardware allocation
- Storage backends
- Monitoring systems
- Training hyperparameters
- Lifecycle management (optional)

**Format**: YAML (`.yaml` or `.yml`)

**Validation**: All configurations are validated using Pydantic schemas before training starts.

## Required Fields

### job_id
**Type**: `string`
**Required**: Yes
**Pattern**: `^[a-zA-Z0-9_-]+$` (alphanumeric, dash, underscore only)

Unique identifier for this training job. Used for:
- Logging and monitoring
- Checkpoint identification
- State tracking
- Email notifications

```yaml
job_id: "llama3-8b-finance-qa"
```

**Best Practices**:
- Use descriptive names: `{model}-{dataset}-{purpose}`
- Include version or date: `llama3-v2-2024-01`
- Keep under 100 characters
- Avoid special characters

### model
**Type**: `string`
**Required**: Yes

Model name or path. Supports:
- HuggingFace model IDs: `meta-llama/Llama-3.1-8B`
- Local paths: `/models/my-model`

```yaml
model: "meta-llama/Llama-3.1-8B"
```

**Popular Models**:
- `meta-llama/Llama-3.1-8B` (8B parameters)
- `meta-llama/Llama-3.1-70B` (70B parameters)
- `mistralai/Mistral-7B-v0.3` (7B parameters)
- `Qwen/Qwen2.5-7B` (7B parameters)
- `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)

### training_type
**Type**: `string`
**Required**: Yes
**Options**: `full`, `lora`, `qlora`

Fine-tuning method to use.

```yaml
training_type: "qlora"  # Most memory-efficient
```

| Type | Description | Memory Usage | Speed | Best For |
|------|-------------|--------------|-------|----------|
| `full` | Full fine-tuning | Highest | Fastest | Small models, many GPUs |
| `lora` | Low-Rank Adaptation | Medium | Fast | 7B-13B models |
| `qlora` | Quantized LoRA (4-bit) | Lowest | Moderate | 70B+ models, limited GPU |

**Recommendations**:
- **1B-3B models**: Use `lora` or `full`
- **7B-13B models**: Use `qlora` or `lora`
- **70B+ models**: Use `qlora` (required)

### nodes
**Type**: `integer`
**Required**: No (default: `1`)
**Range**: `1-256`

Number of compute nodes to use.

```yaml
nodes: 1  # Single node
```

**When to use multiple nodes**:
- Model doesn't fit on single node
- Need more compute for faster training
- Training very large models (70B+)

### gpus_per_node
**Type**: `integer`
**Required**: No (default: `1`)
**Range**: `1-8`

Number of GPUs per node.

```yaml
gpus_per_node: 4
```

**Hardware Limits**:
- Most servers: 1-8 GPUs
- Consumer: 1-2 GPUs
- Cloud instances: 1, 2, 4, or 8 GPUs

**Total GPUs** = `nodes × gpus_per_node` (max 256)

### dataset_type
**Type**: `string`
**Required**: No (default: `"alpaca"`)

Dataset format/prompt strategy.

```yaml
dataset_type: "alpaca"
```

**Common Types**:
- `alpaca` - Alpaca format (instruction, output)
- `sharegpt` - ShareGPT conversation format
- `completion` - Raw completion
- `chat_template` - Uses model's chat template

See [Dataset Formats](DATASETS.md) for details.

### output_dir
**Type**: `string`
**Required**: No (default: `"/workspace/output"`)

Local directory for training outputs.

```yaml
output_dir: "/workspace/output"
```

**Contains**:
- Model checkpoints
- Final model weights
- Training logs
- TensorBoard logs

---

## Storage Configuration

**Required**: Yes

Defines where datasets and models are stored.

### Basic Structure

```yaml
storage:
  type: "s3"  # or azure, gcs, filesystem, hf
  credentials:
    # Backend-specific credentials
  dataset_uri: "s3://bucket/dataset.jsonl"
  output_uri: "s3://bucket/models/output"
  checkpoint_uri: "s3://bucket/checkpoints"  # Optional
```

### storage.type
**Type**: `string`
**Required**: Yes
**Options**: `s3`, `azure`, `gcs`, `filesystem`, `hf`, `huggingface`

Storage backend type.

### storage.dataset_uri
**Type**: `string` (URI)
**Required**: Yes

Location of training dataset.

**Format by backend**:
- **S3**: `s3://bucket/path/to/data.jsonl`
- **Azure**: `azure://container/path/to/data.jsonl`
- **GCS**: `gs://bucket/path/to/data.jsonl`
- **HuggingFace**: `hf://username/dataset-name`
- **Filesystem**: `/data/dataset.jsonl` or `file:///data/dataset.jsonl`

### storage.output_uri
**Type**: `string` (URI)
**Required**: No

Where to upload final trained model.

**Recommendation**: Always specify for production to ensure model is preserved.

### storage.checkpoint_uri
**Type**: `string` (URI)
**Required**: No

Where to store training checkpoints.

**Recommendation**: Essential for long training jobs and multi-day training.

### S3 Storage Example

```yaml
storage:
  type: "s3"
  credentials:
    aws_access_key_id: "${AWS_ACCESS_KEY_ID}"  # Use env vars
    aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
    region: "us-east-1"
  dataset_uri: "s3://my-bucket/datasets/alpaca.jsonl"
  output_uri: "s3://my-bucket/models/llama3-8b-output"
  checkpoint_uri: "s3://my-bucket/checkpoints/llama3-8b"
```

**See**: [Storage Configuration Guide](STORAGE.md) for all backends.

---

## Monitoring Configuration

**Required**: No (but recommended)

Configure monitoring and experiment tracking.

```yaml
monitoring:
  wandb:
    project: "my-project"
    entity: "my-team"
  mlflow:
    tracking_uri: "https://mlflow.example.com"
  tensorboard:
    log_dir: "./runs"
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "training"
    collection: "metrics"
```

### Weights & Biases (W&B)

```yaml
monitoring:
  wandb:
    project: "llama-finetuning"  # Required
    entity: "my-organization"    # Optional
    run_name: "experiment-1"     # Optional (defaults to job_id)
    tags: ["llama", "qlora"]     # Optional
    notes: "Finance QA dataset"  # Optional
```

**Setup**:
1. Create account at https://wandb.ai
2. Get API key: `wandb login`
3. Set `WANDB_API_KEY` environment variable

### MLflow

```yaml
monitoring:
  mlflow:
    tracking_uri: "https://mlflow.example.com"
    experiment_name: "llama-training"
    run_name: "run-001"
```

### TensorBoard

```yaml
monitoring:
  tensorboard:
    log_dir: "./runs"  # Local directory
```

**View logs**: `tensorboard --logdir ./runs`

### MongoDB (Custom Metrics)

```yaml
monitoring:
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "training"
    collection: "metrics"
```

**See**: [Monitoring Guide](MONITORING.md) for setup instructions.

---

## Hyperparameters

**Required**: No (uses sensible defaults)

Training hyperparameters. If not specified, system uses optimized defaults.

```yaml
hyperparameters:
  # Batch size configuration
  batch_size: 128                    # Target effective batch size
  micro_batch_size: 2                # Per-GPU batch size
  gradient_accumulation_steps: 16    # Accumulation steps

  # Learning rate
  learning_rate: 2e-4                # Learning rate
  lr_scheduler: "cosine"             # Scheduler type
  warmup_ratio: 0.1                  # Warmup proportion
  weight_decay: 0.01                 # Weight decay

  # Training duration
  num_epochs: 3                      # Number of epochs

  # Sequence length
  sequence_length: 2048              # Max sequence length

  # LoRA parameters (for lora/qlora only)
  lora_r: 32                         # LoRA rank
  lora_alpha: 64                     # LoRA alpha
  lora_dropout: 0.05                 # LoRA dropout

  # Optimization
  optimizer: "paged_adamw_8bit"      # Optimizer type

  # Evaluation
  val_set_size: 0.05                 # Validation split (5%)
  evals_per_epoch: 2                 # Evaluations per epoch
  saves_per_epoch: 1                 # Checkpoints per epoch
  logging_steps: 10                  # Log every N steps

  # Advanced (optional)
  fsdp_offload: false                # CPU offloading for large models
```

### Key Hyperparameters Explained

#### Batch Sizes

**Effective Batch Size** = `micro_batch_size × gradient_accumulation_steps × total_gpus`

```yaml
# Example: 4 GPUs
micro_batch_size: 2
gradient_accumulation_steps: 16
# Effective batch = 2 × 16 × 4 = 128
```

**Recommendations**:
- **micro_batch_size**: Start with 1-2, increase until OOM
- **gradient_accumulation_steps**: Adjust to reach target batch size
- **Target effective batch**: 64-256 for most LLMs

#### Learning Rate

```yaml
learning_rate: 2e-4  # 0.0002
```

**Typical Ranges**:
- Full fine-tuning: `1e-5` to `5e-5`
- LoRA: `1e-4` to `3e-4`
- QLoRA: `2e-4` to `5e-4`

**Rule of thumb**: Smaller models can use higher learning rates.

#### LoRA Parameters

```yaml
lora_r: 32        # Rank (4, 8, 16, 32, 64)
lora_alpha: 64    # Alpha (usually 2× rank)
lora_dropout: 0.05
```

**Guidelines**:
- Higher `lora_r` = more capacity but slower/more memory
- Typical values: `r=16-32`, `alpha=32-64`
- Start with `r=32` for most tasks

### Optimizer Options

```yaml
optimizer: "paged_adamw_8bit"
```

**Available Optimizers**:
- `paged_adamw_8bit` - Memory-efficient (recommended for QLoRA)
- `adamw_8bit` - 8-bit AdamW
- `adamw_torch_fused` - Fastest, full precision
- `adamw_torch` - Standard AdamW
- `sgd` - SGD optimizer

**See**: [Hyperparameters Guide](HYPERPARAMETERS.md) for detailed tuning.

---

## Lifecycle Configuration

**Required**: No (advanced feature)

Enterprise features for production deployments.

```yaml
lifecycle:
  # State tracking (MongoDB-based)
  state_tracking:
    enabled: true
    mongodb:
      uri: "mongodb://localhost:27017"
      database: "training_jobs"
      collection: "job_states"

  # Heartbeat monitoring
  heartbeat:
    enabled: true
    interval_seconds: 60      # Send heartbeat every 60s
    timeout_seconds: 3600     # Mark as timeout after 1 hour

  # Automatic retry
  retry:
    enabled: true
    max_attempts: 3
    delay_minutes: [5, 10, 20]  # Increasing delays

  # Email notifications
  notifications:
    email:
      enabled: true
      from: "training@example.com"
      to: ["team@example.com"]
      smtp:
        server: "smtp.gmail.com"
        port: 587
        use_tls: true
        # Credentials from secrets manager (recommended)
        secret_path: "training/smtp"

  # Webhooks (for server deletion, etc.)
  server_deletion:
    enabled: true
    webhook:
      url: "https://api.example.com/delete-server"
      method: "POST"
      headers:
        Authorization: "Bearer ${API_TOKEN}"
    delete_on_success:
      enabled: true
      delay_minutes: 5
    delete_on_failure:
      enabled: true
      delay_hours: 1
```

**See**: [Lifecycle Management](LIFECYCLE.md) for details.

---

## Secrets Management

**Required**: No (but highly recommended for production)

Secure credential management instead of hardcoding.

```yaml
secrets_manager:
  type: "vault"  # or "aws", "env"
  config:
    vault_addr: "https://vault.example.com:8200"
    vault_token: "${VAULT_TOKEN}"
    mount_point: "secret"
```

**Then reference secrets**:
```yaml
storage:
  type: "s3"
  secret_path: "aws/training-credentials"  # Instead of inline creds
```

**See**: [Secrets Management](SECRETS.md) for setup.

---

## Auto-Configuration

**New in v3.0**: Automatic hyperparameter optimization.

```yaml
auto_config:
  enabled: true
  optimize_for: "balanced"  # speed, memory, quality, balanced

  # User MUST provide these (dataset-specific):
  sequence_length: 2048
  sample_packing: true
  pad_to_sequence_len: true
```

**How it works**:
1. Analyzes your model (parameters, architecture)
2. Detects your hardware (GPU type, memory, count)
3. Uses your dataset parameters
4. Calculates optimal batch sizes, parallelism, etc.

**Optimization Modes**:
- `speed` - Maximize throughput (larger batches, aggressive settings)
- `memory` - Minimize memory usage (small batches, offloading)
- `quality` - Best model quality (conservative settings)
- `balanced` - Good balance of all three (**recommended**)

**See**: [Auto-Optimizer Guide](AUTO_OPTIMIZER.md)

---

## Complete Example

### Production QLoRA Training (70B Model)

```yaml
# Job identification
job_id: "llama3-70b-finance-v1"
model: "meta-llama/Llama-3.1-70B"
training_type: "qlora"

# Hardware
nodes: 4
gpus_per_node: 8

# Dataset
dataset_type: "alpaca"

# Storage (S3)
storage:
  type: "s3"
  secret_path: "aws/training"  # Credentials from secrets manager
  dataset_uri: "s3://training-data/finance-qa.jsonl"
  output_uri: "s3://models/llama3-70b-finance"
  checkpoint_uri: "s3://checkpoints/llama3-70b-finance"

# Monitoring
monitoring:
  wandb:
    project: "llama-finance"
    entity: "ml-team"
    tags: ["70b", "qlora", "finance"]
  mongodb:
    uri: "mongodb://mongo.internal:27017"
    database: "training"
    collection: "metrics"

# Hyperparameters
hyperparameters:
  micro_batch_size: 1
  gradient_accumulation_steps: 32
  learning_rate: 1e-4
  num_epochs: 2
  sequence_length: 4096
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  optimizer: "paged_adamw_8bit"
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  val_set_size: 0.05
  evals_per_epoch: 4
  saves_per_epoch: 2
  logging_steps: 10

# Lifecycle (production features)
lifecycle:
  state_tracking:
    enabled: true
    mongodb:
      uri: "mongodb://mongo.internal:27017"
      database: "training_jobs"
      collection: "states"

  heartbeat:
    enabled: true
    interval_seconds: 60
    timeout_seconds: 3600

  retry:
    enabled: true
    max_attempts: 3
    delay_minutes: [10, 30, 60]

  notifications:
    email:
      enabled: true
      from: "ml-training@company.com"
      to: ["team@company.com", "oncall@company.com"]
      smtp:
        server: "smtp.company.com"
        port: 587
        use_tls: true
        secret_path: "smtp/credentials"

# Secrets
secrets_manager:
  type: "vault"
  config:
    vault_addr: "https://vault.company.com:8200"
    vault_token: "${VAULT_TOKEN}"

# Output
output_dir: "/workspace/output"
```

### Simple Development Example

```yaml
job_id: "test-training"
model: "NousResearch/Llama-3.2-1B"
training_type: "qlora"

nodes: 1
gpus_per_node: 1

storage:
  type: "filesystem"
  dataset_uri: "/data/my_dataset.jsonl"
  output_uri: "/models/output"

monitoring:
  tensorboard:
    log_dir: "./runs"

hyperparameters:
  micro_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  num_epochs: 1
  sequence_length: 512
```

---

## Validation

All configurations are validated before training:

```bash
# Validate configuration without training
python integration/train_wrapper.py \
  --job-config my_job.yaml \
  --validate-only
```

**Common Validation Errors**:
1. Invalid job_id format
2. Missing required fields
3. Invalid storage URIs
4. Incompatible training settings
5. Resource limits exceeded

**See**: [Troubleshooting](TROUBLESHOOTING.md#configuration-errors)

---

## Next Steps

- **[Hyperparameters Guide](HYPERPARAMETERS.md)** - Detailed parameter tuning
- **[Storage Configuration](STORAGE.md)** - Setup storage backends
- **[Monitoring Setup](MONITORING.md)** - Configure tracking
- **[Examples](EXAMPLES.md)** - More complete examples
