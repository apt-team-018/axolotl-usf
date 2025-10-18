# Troubleshooting Guide

Solutions to common issues and errors.

## Table of Contents
- [Out of Memory (OOM)](#out-of-memory-oom)
- [Configuration Errors](#configuration-errors)
- [Storage Issues](#storage-issues)
- [Training Failures](#training-failures)
- [Multi-Node Issues](#multi-node-issues)
- [Performance Problems](#performance-problems)
- [Monitoring Issues](#monitoring-issues)
- [Installation Problems](#installation-problems)

---

## Out of Memory (OOM)

### CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:

#### 1. Reduce Batch Size

```yaml
hyperparameters:
  micro_batch_size: 1  # Reduce from 2→1
  gradient_accumulation_steps: 64  # Increase to compensate
```

#### 2. Enable Gradient Checkpointing

```yaml
hyperparameters:
  gradient_checkpointing: true
```

Saves ~30-40% memory at cost of ~20% speed.

#### 3. Reduce Sequence Length

```yaml
hyperparameters:
  sequence_length: 1024  # Reduce from 2048
```

Memory usage ∝ sequence_length²

#### 4. Switch to QLoRA

```yaml
training_type: "qlora"  # Instead of "lora" or "full"
```

QLoRA uses 4-bit quantization, saving ~60% memory.

#### 5. Enable FSDP Offloading

```yaml
hyperparameters:
  fsdp_offload: true
```

Offloads to CPU RAM (slower but prevents OOM).

#### 6. Use 8-bit Optimizer

```yaml
hyperparameters:
  optimizer: "paged_adamw_8bit"  # Instead of adamw_torch
```

#### 7. Use Auto-Optimizer (Memory Mode)

```yaml
auto_config:
  enabled: true
  optimize_for: "memory"
  sequence_length: 2048
  sample_packing: true
```

**Memory Hierarchy** (least → most memory):
1. QLoRA + 8-bit optimizer + gradient checkpointing + offloading
2. QLoRA + 8-bit optimizer + gradient checkpointing
3. LoRA + 8-bit optimizer + gradient checkpointing
4. Full fine-tuning

---

## Configuration Errors

### Invalid job_id

**Error**:
```
ValidationError: job_id must contain only alphanumeric characters, dashes, and underscores
```

**Solution**:
```yaml
# Bad
job_id: "my training job"  # Spaces not allowed
job_id: "job@2024"         # @ not allowed

# Good
job_id: "my-training-job"
job_id: "job_2024"
job_id: "training-v1"
```

### Missing Required Fields

**Error**:
```
ValidationError: field required: storage
```

**Solution**:
```yaml
# Add all required fields:
job_id: "my-job"
model: "meta-llama/Llama-3.1-8B"
training_type: "qlora"

storage:
  type: "s3"
  dataset_uri: "s3://bucket/data.jsonl"
```

### Invalid URI Format

**Error**:
```
Invalid URI format for dataset_uri: bucket/data.jsonl (missing scheme)
```

**Solution**:
```yaml
# Bad
dataset_uri: "bucket/data.jsonl"

# Good
dataset_uri: "s3://bucket/data.jsonl"
dataset_uri: "/data/dataset.jsonl"
dataset_uri: "file:///data/dataset.jsonl"
```

### Learning Rate Too High

**Error**:
```
ValidationError: Learning rate 0.1 is very high. Typical range: 1e-5 to 1e-3
```

**Solution**:
```yaml
# Too high
learning_rate: 0.1

# Correct
learning_rate: 2e-4  # 0.0002
```

### Validation: Before Training

**Always validate first**:
```bash
python integration/train_wrapper.py \
  --job-config my_job.yaml \
  --validate-only
```

---

## Storage Issues

### S3 Access Denied

**Error**:
```
ClientError: An error occurred (AccessDenied) when calling GetObject
```

**Solutions**:

#### 1. Check IAM Permissions

Required permissions:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::my-bucket/*",
    "arn:aws:s3:::my-bucket"
  ]
}
```

#### 2. Verify Credentials

```bash
# Test AWS credentials
aws s3 ls s3://my-bucket/

# Test with Python
python3 -c "import boto3; s3=boto3.client('s3'); print(s3.list_buckets())"
```

#### 3. Check Bucket Policy

Ensure bucket allows your account/role.

### Connection Timeout

**Error**:
```
ReadTimeoutError: Read timed out
```

**Solutions**:

#### 1. Check Network

```bash
# Test connectivity
ping s3.amazonaws.com

# Test with curl
curl -I https://my-bucket.s3.amazonaws.com/
```

#### 2. Increase Timeout

```yaml
storage:
  type: "s3"
  config:
    connect_timeout: 60
    read_timeout: 300
```

#### 3. Use Retry

System automatically retries (5 attempts), but you can adjust:
```python
# In custom backend
@retry_on_exception(max_attempts=10, min_wait=10, max_wait=120)
def download_dataset(...):
    ...
```

### Checksum Verification Failed

**Error**:
```
RuntimeError: Download checksum verification failed
```

**Solutions**:

1. **Re-upload dataset** (may be corrupted)
2. **Check network stability**
3. **Disable checksum** (not recommended):
```python
# Only for testing
# storage_manager.verify_checksums = False
```

### Disk Space

**Error**:
```
RuntimeError: Insufficient disk space at /workspace/data: 5.2GB available, 100GB required
```

**Solutions**:

#### 1. Clean Up

```bash
# Remove old checkpoints
rm -rf /workspace/checkpoints/*

# Remove old outputs
rm -rf /workspace/output/*

# Check disk usage
df -h /workspace
```

#### 2. Increase Disk Size

- AWS: Modify EBS volume size
- GCP: Resize disk
- Local: Add more storage

#### 3. Stream Dataset

For very large datasets, stream instead of downloading (advanced).

---

## Training Failures

### Loss is NaN

**Error**:
```
Step 50: loss=nan
```

**Causes & Solutions**:

#### 1. Learning Rate Too High

```yaml
# Reduce learning rate
hyperparameters:
  learning_rate: 1e-5  # From 1e-4
```

#### 2. Mixed Precision Issues

```yaml
# Disable bf16/fp16
hyperparameters:
  bf16: false
  fp16: false
```

#### 3. Bad Initialization

```bash
# Restart with different seed
SEED=42 python integration/train_wrapper.py --job-config job.yaml
```

#### 4. Data Issues

Check for:
- Inf or NaN in dataset
- Extremely large values
- Empty sequences

### Training Stuck (No Progress)

**Symptoms**: Loss not decreasing after many steps

**Solutions**:

#### 1. Increase Learning Rate

```yaml
hyperparameters:
  learning_rate: 2e-4  # From 1e-5
```

#### 2. Check Gradients

```yaml
# Enable gradient clipping
hyperparameters:
  gradient_clipping: 1.0
```

#### 3. Verify Data Loading

```python
# Test dataset loading
from datasets import load_dataset
ds = load_dataset("json", data_files="data.jsonl")
print(ds[0])  # Should show actual data
```

### Checkpoint Save Failed

**Error**:
```
Failed to upload checkpoint: ConnectionError
```

**Solutions**:

#### 1. Check Storage Connectivity

```bash
# Test upload
aws s3 cp test.txt s3://bucket/test.txt
```

#### 2. Increase Timeout

Storage operations have built-in retry, but ensure network is stable.

#### 3. Use Local Checkpoints

```yaml
# Temporarily save locally
storage:
  checkpoint_uri: "/workspace/checkpoints"
```

### Process Killed (OOM Killer)

**Error**:
```
Killed
```

**Cause**: System ran out of RAM (not GPU memory)

**Solutions**:

#### 1. Enable FSDP CPU Offloading

```yaml
hyperparameters:
  fsdp_offload: true
```

#### 2. Reduce Workers

```yaml
hyperparameters:
  dataloader_num_workers: 0  # From 4
```

#### 3. Increase System RAM

Or use instance with more RAM.

---

## Multi-Node Issues

### NCCL Timeout

**Error**:
```
RuntimeError: NCCL timeout after 1800s
```

**Solutions**:

#### 1. Increase Timeout

```bash
export NCCL_TIMEOUT=7200  # 2 hours
```

#### 2. Check Network

```bash
# Test inter-node connectivity
ping <other-node-ip>

# Test bandwidth
iperf3 -s  # On one node
iperf3 -c <node1-ip>  # On another
```

#### 3. Verify NCCL Settings

```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Or ib0 for InfiniBand
```

### Rank Mismatch

**Error**:
```
RuntimeError: Default process group has not been initialized
```

**Solutions**:

#### 1. Check Environment Variables

Ensure same on all nodes:
```bash
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500
export WORLD_SIZE=8  # Total GPUs
export RANK=0  # Different on each node (0,1,2...)
```

#### 2. Verify torchrun Arguments

```bash
# On each node, ensure consistent:
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \  # 0 on first, 1 on second
  --master_addr=<ip> \
  --master_port=29500
```

### Distributed Initialization Failed

**Error**:
```
RuntimeError: Distributed package doesn't have NCCL built in
```

**Solution**:

Rebuild PyTorch with NCCL:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Performance Problems

### Slow Training

**Symptoms**: Very low tokens/second

**Diagnostics**:

```bash
# Monitor GPU usage
nvidia-smi dmon -i 0

# Should show 90-100% GPU utilization
```

**Solutions**:

#### 1. Increase Batch Size

```yaml
hyperparameters:
  micro_batch_size: 4  # From 1
```

#### 2. Enable Flash Attention

```yaml
hyperparameters:
  flash_attention: true
```

#### 3. Use Sample Packing

```yaml
hyperparameters:
  sample_packing: true
```

#### 4. Reduce Sequence Length

```yaml
hyperparameters:
  sequence_length: 2048  # From 4096
```

#### 5. Disable Unnecessary Logging

```yaml
hyperparameters:
  logging_steps: 100  # From 1
```

#### 6. Use Auto-Optimizer (Speed Mode)

```yaml
auto_config:
  enabled: true
  optimize_for: "speed"
```

### Low GPU Utilization

**Symptoms**: GPU at 50% or less

**Causes & Solutions**:

#### 1. CPU Bottleneck

```yaml
hyperparameters:
  dataloader_num_workers: 8  # Increase from 4
```

#### 2. Disk I/O Bottleneck

- Use faster storage (NVMe SSD)
- Preload dataset to RAM
- Use local storage instead of network

#### 3. Small Batch Size

Increase `micro_batch_size` until GPU utilization is 90%+.

---

## Monitoring Issues

### W&B Authentication Failed

**Error**:
```
wandb.errors.UsageError: api_key not configured
```

**Solution**:

```bash
# Login to W&B
wandb login

# Or set API key
export WANDB_API_KEY=<your-key>
```

### MongoDB Connection Failed

**Error**:
```
pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 61] Connection refused
```

**Solutions**:

#### 1. Start MongoDB

```bash
# Linux
sudo systemctl start mongod

# Docker
docker run -d -p 27017:27017 mongo:latest
```

#### 2. Check Connection String

```yaml
monitoring:
  mongodb:
    uri: "mongodb://localhost:27017"  # Not "localhost" without protocol
```

#### 3. Allow Network Access

```bash
# Check MongoDB is listening
netstat -an | grep 27017
```

### TensorBoard Not Showing Logs

**Problem**: TensorBoard empty or outdated

**Solutions**:

#### 1. Check Log Directory

```bash
ls -la ./runs/
```

Should show timestamped directories with `events.out.tfevents.*` files.

#### 2. Refresh TensorBoard

```bash
# Stop and restart
tensorboard --logdir ./runs --reload_interval 10
```

#### 3. Verify Configuration

```yaml
monitoring:
  tensorboard:
    log_dir: "./runs"  # Must match --logdir
```

---

## Installation Problems

### CUDA Not Found

**Error**:
```
RuntimeError: CUDA not available
```

**Solutions**:

#### 1. Verify CUDA Installation

```bash
nvidia-smi
nvcc --version
```

#### 2. Install CUDA Toolkit

```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
```

#### 3. Reinstall PyTorch

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'axolotl'
```

**Solution**:

```bash
# Install Axolotl
cd /path/to/axolotl-usf
pip install -e .

# Install integration requirements
pip install -r integration/requirements.txt
```

### Flash Attention Build Failed

**Error**:
```
ERROR: Failed building wheel for flash-attn
```

**Solutions**:

#### 1. Install Pre-built Wheel

```bash
pip install flash-attn --no-build-isolation
```

#### 2. Use Pre-compiled

```bash
pip install flash-attn==2.5.0  # Specific version
```

#### 3. Disable Flash Attention

```yaml
hyperparameters:
  flash_attention: false
```

---

## Getting More Help

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
python integration/train_wrapper.py --job-config job.yaml
```

### Check Logs

```bash
# View recent logs
tail -f /workspace/logs/training.log

# Search for errors
grep -i "error" /workspace/logs/training.log
```

### Test Configuration

```bash
# Validate without training
python integration/train_wrapper.py \
  --job-config job.yaml \
  --validate-only
```

### Minimal Reproduction

Create minimal config that reproduces the issue:

```yaml
job_id: "debug-test"
model: "NousResearch/Llama-3.2-1B"  # Small model
training_type: "qlora"
nodes: 1
gpus_per_node: 1

storage:
  type: "filesystem"
  dataset_uri: "/data/tiny-dataset.jsonl"  # Tiny dataset
  output_uri: "/tmp/output"

hyperparameters:
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  num_epochs: 1
  max_steps: 10  # Only 10 steps for testing
```

### Report Issues

When reporting issues, include:
1. **Full error message and stack trace**
2. **Job configuration** (sanitize credentials)
3. **System info**: GPU type, CUDA version, PyTorch version
4. **Logs**: Debug logs around error
5. **Reproducibility**: Minimal config that reproduces issue

---

## Quick Reference

### Common Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Network/retryable error |
| 3 | Out of memory |
| 4 | Configuration error |
| 130 | Ctrl+C (SIGINT) |

### Health Check Commands

```bash
# Check GPU
nvidia-smi

# Check Python environment
python -c "import torch; print(torch.cuda.is_available())"

# Check storage
aws s3 ls s3://my-bucket/  # S3
az storage blob list  # Azure

# Check network
ping google.com
```

### Quick Fixes Checklist

- [ ] Validated configuration: `--validate-only`
- [ ] Checked GPU memory: `nvidia-smi`
- [ ] Verified dataset accessible
- [ ] Checked credentials/permissions
- [ ] Reviewed logs for errors
- [ ] Tried with smaller model/batch
- [ ] Disabled optional features (flash attn, etc.)
- [ ] Tested with minimal configuration

---

## Next Steps

- **[Performance Tuning](PERFORMANCE.md)** - Optimization strategies
- **[Job Configuration](JOB_CONFIG.md)** - Configuration reference
- **[Examples](EXAMPLES.md)** - Working configurations
- **[FAQ](FAQ.md)** - Frequently asked questions
