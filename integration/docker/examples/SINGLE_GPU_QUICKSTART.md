# Single GPU QuickStart Guide

Complete guide for running LoRA fine-tuning on a single GPU using cloud platforms (RunPod, Vast.ai, Azure, AWS) or local machines.

## 📋 Overview

**Configuration**: Qwen2.5-0.5B-Instruct + LoRA + 100 HuggingFace samples
- **Model**: Qwen/Qwen2.5-0.5B-Instruct (small, fast for testing)
- **Dataset**: HuggingFace dataset (mlabonne/FineTome-100k, using 100 samples)
- **Training**: LoRA (low resource requirements)
- **Time**: ~5-10 minutes on H100/H200
- **GPU Memory**: ~8-12GB

## 🚀 Quick Start (3 Commands)

```bash
# 1. Build the Docker image
bash integration/docker/build.sh

# 2. Run training with test config
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:latest

# 3. Check the output
ls -lh output/
```

## 🌐 Platform-Specific Guides

### Option 1: RunPod

**Step 1**: Deploy a GPU instance
- Go to [RunPod](https://www.runpod.io/)
- Choose GPU: H100, A100, or H200
- Template: Use "RunPod PyTorch" or similar
- Deploy instance

**Step 2**: SSH into instance and setup
```bash
# SSH into your RunPod instance
ssh root@<runpod-ip> -p <port>

# Clone your repository (or use RunPod's volume)
git clone https://github.com/your-org/axolotl-usf.git
cd axolotl-usf

# Build Docker image
bash integration/docker/build.sh

# Run training
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  -v /runpod-volume/cache:/root/.cache/huggingface \
  axolotl-usf:latest
```

**Step 3**: Monitor training
```bash
# View logs
docker logs -f <container-id>

# Check GPU usage
nvidia-smi -l 1
```

**Step 4**: Download results
```bash
# From your local machine
scp -P <port> root@<runpod-ip>:/root/axolotl-usf/output/* ./local-output/
```

### Option 2: Vast.ai

**Step 1**: Rent a GPU
- Go to [Vast.ai](https://vast.ai/)
- Search for: H100, A100, or RTX 4090
- Filter: Docker support
- Rent instance

**Step 2**: Use Docker image directly
```bash
# SSH into Vast.ai instance
ssh root@<vast-ip> -p <port>

# Pull pre-built image (or build locally)
docker pull axolotl-usf:latest

# Run training
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v /workspace/output:/workspace/output \
  axolotl-usf:latest
```

### Option 3: Azure VM with GPU

**Step 1**: Create GPU VM
```bash
# Create resource group
az group create --name axolotl-rg --location eastus

# Create GPU VM (NC6s_v3 has 1x V100)
az vm create \
  --resource-group axolotl-rg \
  --name axolotl-vm \
  --image Canonical:UbuntuServer:20.04-LTS:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

**Step 2**: Setup VM
```bash
# SSH into VM
ssh azureuser@<vm-ip>

# Install Docker and NVIDIA runtime
curl -fsSL https://get.docker.com | sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone and build
git clone https://github.com/your-org/axolotl-usf.git
cd axolotl-usf
bash integration/docker/build.sh
```

**Step 3**: Run training
```bash
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:latest
```

### Option 4: AWS EC2 with GPU

**Step 1**: Launch EC2 instance
```bash
# Launch p3.2xlarge (1x V100) or g5.xlarge (1x A10G)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \  # Deep Learning AMI
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx
```

**Step 2**: Setup and run
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<ec2-ip>

# Docker and NVIDIA runtime already installed in Deep Learning AMI
git clone https://github.com/your-org/axolotl-usf.git
cd axolotl-usf
bash integration/docker/build.sh

# Run training
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:latest
```

### Option 5: Local Machine with GPU

**Prerequisites**: Docker with NVIDIA runtime installed

```bash
# Navigate to project
cd /path/to/axolotl-usf

# Build image
bash integration/docker/build.sh

# Run training
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  axolotl-usf:latest
```

## 📝 Configuration Explained

The test configuration [`single-gpu-lora-test.yaml`](single-gpu-lora-test.yaml) uses:

```yaml
# Small model for fast testing
model: "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA training (low memory)
training_type: "lora"
hyperparameters:
  lora_r: 32
  lora_alpha: 64

# Small batch for single GPU
micro_batch_size: 4
gradient_accumulation_steps: 4

# Quick test (500 steps)
max_steps: 500
```

## 🔧 Customizing for Your Use Case

### Use Your Own HuggingFace Dataset

Edit the config file:
```yaml
storage:
  dataset_uri: "hf://your-username/your-dataset"
```

### Use Local Dataset (100 samples)

1. Create dataset file `my_data.jsonl`:
```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
{"messages": [{"role": "user", "content": "Explain ML"}, {"role": "assistant", "content": "ML is..."}]}
```

2. Update config:
```yaml
storage:
  dataset_uri: "/workspace/data/my_data.jsonl"
```

3. Mount the file:
```bash
docker run --gpus all \
  -v $(pwd)/my_data.jsonl:/workspace/data/my_data.jsonl \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:latest
```

### Use Larger Model (requires more memory)

For Llama-3.1-8B or similar:
```yaml
model: "meta-llama/Llama-3.1-8B-Instruct"
hyperparameters:
  micro_batch_size: 1  # Reduce batch size
  gradient_accumulation_steps: 16  # Increase accumulation
```

### Train Longer

```yaml
hyperparameters:
  num_epochs: 3  # Instead of max_steps
  # Or
  max_steps: 5000  # More steps
```

## 📊 Expected Results

### Training Progress
```
Step 10/500 | Loss: 2.456 | LR: 0.0001
Step 50/500 | Loss: 1.234 | LR: 0.00018
Step 100/500 | Loss: 0.876 | LR: 0.0002
...
Step 500/500 | Loss: 0.345 | LR: 0.00005
```

### Output Files
```
output/
├── adapter_config.json
├── adapter_model.safetensors
├── README.md
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### Model Size
- **LoRA Adapters**: ~50-100MB (vs 1-7GB for full model)
- **Full Model (if merged)**: Same as original model

## 🎯 Performance Expectations

| GPU | Model | Batch Size | Time (500 steps) | Memory Usage |
|-----|-------|------------|------------------|--------------|
| H200 | Qwen2.5-0.5B | 4 | ~3-5 min | ~6-8GB |
| H100 | Qwen2.5-0.5B | 4 | ~4-6 min | ~6-8GB |
| A100 | Qwen2.5-0.5B | 4 | ~6-8 min | ~8-10GB |
| RTX 4090 | Qwen2.5-0.5B | 4 | ~8-10 min | ~8-10GB |
| V100 | Qwen2.5-0.5B | 2 | ~12-15 min | ~10-12GB |

## 🔍 Monitoring Training

### View Logs in Real-Time
```bash
# Get container ID
docker ps

# Follow logs
docker logs -f <container-id>
```

### Check GPU Usage
```bash
# Inside container
docker exec <container-id> nvidia-smi

# From host (every 2 seconds)
watch -n 2 nvidia-smi
```

### Use Monitoring Tools

Add to config file:
```yaml
monitoring:
  wandb:
    project: "my-lora-test"
    api_key: "your-wandb-key"
```

Then view at: https://wandb.ai/your-username/my-lora-test

## 🐛 Troubleshooting

### Out of Memory
**Solution**: Reduce batch size
```yaml
micro_batch_size: 1  # Instead of 4
gradient_accumulation_steps: 16  # Instead of 4
```

### Dataset Not Found
**Solution**: Check HuggingFace dataset exists
```bash
# Test dataset loading
docker run --gpus all -it axolotl-usf:latest python3 -c "
from datasets import load_dataset
ds = load_dataset('mlabonne/FineTome-100k', split='train[:100]')
print(f'Loaded {len(ds)} samples')
"
```

### Slow Download
**Solution**: Use HuggingFace cache
```bash
# Mount HF cache
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ...
```

### Docker GPU Not Detected
**Solution**: Verify NVIDIA runtime
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## ✅ Verification Checklist

Before running training:
- [ ] Docker installed with NVIDIA runtime
- [ ] GPU detected (`nvidia-smi` works)
- [ ] Image built successfully
- [ ] Config file exists
- [ ] Output directory is writable

## 🎓 Next Steps

After successful test:

1. **Scale Up**: Use larger model (Llama, Qwen 7B+)
2. **More Data**: Increase dataset size
3. **Production**: Add monitoring, checkpointing, cloud storage
4. **Multi-GPU**: Try multi-node setup
5. **Deploy**: Use trained model for inference

## 📚 Related Documentation

- [Job Configuration Guide](../../docs/JOB_CONFIG.md)
- [Docker README](../README.md)
- [Multi-Node Setup](MULTI_NODE_EXAMPLE.md)
- [Hyperparameter Tuning](../../docs/HYPERPARAMETERS.md)

---

**Start your first training in under 5 minutes!** 🚀
