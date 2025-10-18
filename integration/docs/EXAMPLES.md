# Complete Configuration Examples

Production-ready configuration examples for different scenarios using the integration system.

## Table of Contents
- [Beginner Examples](#beginner-examples)
- [Intermediate Examples](#intermediate-examples)
- [Production Examples](#production-examples)
- [Specialized Use Cases](#specialized-use-cases)

---

## Beginner Examples

### Example 1: Local Development (1B Model, Single GPU)

**Scenario**: Testing on laptop/workstation with single consumer GPU

```yaml
# local-dev-1b.yaml
job_id: "local-test-1b-qlora"
model: "NousResearch/Llama-3.2-1B"
training_type: "qlora"

# Hardware
nodes: 1
gpus_per_node: 1

# Storage (local filesystem)
storage:
  type: "filesystem"
  dataset_uri: "/Users/username/data/conversations.jsonl"
  output_uri: "/Users/username/models/output"

# Monitoring (TensorBoard only for simplicity)
monitoring:
  tensorboard:
    log_dir: "./runs"

# Hyperparameters (small and fast for testing)
hyperparameters:
  # Dataset
  sequence_len: 512  # Short for quick testing
  sample_packing: true

  # Batch size
  micro_batch_size: 2
  gradient_accumulation_steps: 4

  # Learning
  learning_rate: 3e-4
  num_epochs: 1
  max_steps: 50  # Quick test

  # LoRA
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

  # Precision
  bf16: auto
  gradient_checkpointing: true
  flash_attention: true

  # Evaluation
  val_set_size: 0.1
  logging_steps: 5

output_dir: "/Users/username/output"
```

**Run**:
```bash
python integration/train_wrapper.py --job-config local-dev-1b.yaml
```

---

### Example 2: Cloud Single GPU (7B Model, RunPod/Vast.ai)

**Scenario**: Single A100 (40GB/80GB) on cloud provider

```yaml
# cloud-7b-qlora.yaml
job_id: "cloud-llama-7b-qlora"
model: "meta-llama/Llama-3.1-8B"
training_type: "qlora"

# Hardware
nodes: 1
gpus_per_node: 1

# Storage (HuggingFace for easy access)
storage:
  type: "hf"
  credentials:
    hf_token: "${HF_TOKEN}"
  dataset_uri: "hf://username/training-conversations"
  output_uri: "hf://username/llama-8b-finetuned"

# Monitoring
monitoring:
  wandb:
    project: "llama-finetuning"
    tags: ["8b", "qlora", "cloud"]
  tensorboard:
    log_dir: "/workspace/runs"

# Hyperparameters
hyperparameters:
  # Dataset
  sequence_len: 2048
  sample_packing: true
  pad_to_sequence_len: true
  dataset_num_proc: 4

  # Batch size (maximize for A100-80GB)
  micro_batch_size: 4
  gradient_accumulation_steps: 8
  # Effective batch = 4 × 8 × 1 = 32

  # Learning
  learning_rate: 2e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  num_epochs: 3

  # LoRA
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  lora_target_linear: true

  # Optimizer
  optimizer: "paged_adamw_8bit"

  # Precision
  bf16: auto
  tf32: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  # Performance
  flash_attention: true
  lora_mlp_kernel: true
  lora_qkv_kernel: true
  lora_o_kernel: true

  # Evaluation
  val_set_size: 0.05
  evals_per_epoch: 2
  saves_per_epoch: 1
  save_total_limit: 3
  logging_steps: 10

output_dir: "/workspace/output"
```

---

### Example 3: Auto-Optimized (Beginner-Friendly)

**Scenario**: Let system optimize everything automatically

```yaml
# auto-optimized-7b.yaml
job_id: "auto-llama-7b"
model: "mistralai/Mistral-7B-v0.3"
training_type: "qlora"

# Hardware
nodes: 1
gpus_per_node: 4

# Storage
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://my-bucket/conversations.jsonl"
  output_uri: "s3://my-bucket/models/mistral-7b-output"
  checkpoint_uri: "s3://my-bucket/checkpoints/mistral-7b"

# Monitoring
monitoring:
  wandb:
    project: "mistral-experiments"
    entity: "ml-team"

# Auto-Config (system calculates optimal settings)
auto_config:
  enabled: true
  optimize_for: "balanced"
  sequence_length: 2048  # YOU provide this
  sample_packing: true   # YOU provide this

# Manual overrides (optional)
hyperparameters:
  learning_rate: 2e-4
  num_epochs: 3
  lora_r: 32
  lora_alpha: 64

output_dir: "/workspace/output"
```

**What auto-config sets automatically**:
- `micro_batch_size: 4`
- `gradient_accumulation_steps: 8`
- `optimizer: "adamw_torch_fused"`
- `flash_attention: true`
- `bf16: true`
- `gradient_checkpointing: true`
- And more...

---

## Intermediate Examples

### Example 4: Multi-GPU Single Node (13B Model)

**Scenario**: 4× A100 (80GB) on single server

```yaml
# multi-gpu-13b.yaml
job_id: "llama-13b-4gpu-qlora"
model: "meta-llama/Llama-2-13b-hf"
training_type: "qlora"

# Hardware
nodes: 1
gpus_per_node: 4

# Storage
storage:
  type: "s3"
  secret_path: "aws/training-credentials"
  dataset_uri: "s3://training-data/customer-support.jsonl"
  output_uri: "s3://models/llama-13b-support"
  checkpoint_uri: "s3://checkpoints/llama-13b-support"

# Monitoring (multiple backends)
monitoring:
  wandb:
    project: "llama-13b-support"
    entity: "ml-team"
    tags: ["13b", "support", "qlora"]
  mlflow:
    tracking_uri: "https://mlflow.company.com"
    experiment_name: "customer-support"
  mongodb:
    uri: "mongodb://mongo:27017"
    database: "training"
    collection: "metrics"

# Hyperparameters
hyperparameters:
  # Dataset
  sequence_len: 4096
  sample_packing: true
  pad_to_sequence_len: true
  dataset_num_proc: 16

  # Batch size
  micro_batch_size: 2
  gradient_accumulation_steps: 16
  # Effective batch = 2 × 16 × 4 = 128

  # Learning
  learning_rate: 1e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  num_epochs: 3

  # LoRA
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  lora_target_linear: true
  peft_use_rslora: false

  # Optimizer
  optimizer: "paged_adamw_8bit"

  # Precision
  bf16: auto
  tf32: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  # Performance
  flash_attention: true
  lora_mlp_kernel: true
  lora_qkv_kernel: true
  lora_o_kernel: true

  # Evaluation
  val_set_size: 0.05
  evals_per_epoch: 4
  saves_per_epoch: 2
  save_total_limit: 3
  logging_steps: 10
  early_stopping_patience: 3

  # Advanced
  loss_watchdog_threshold: 5.0
  loss_watchdog_patience: 3

output_dir: "/workspace/output"

# Secrets management
secrets_manager:
  type: "vault"
  config:
    vault_addr: "https://vault.company.com:8200"
    vault_token: "${VAULT_TOKEN}"
```

---

## Production Examples

### Example 5: Production 70B Multi-Node (Enterprise)

**Scenario**: 8 nodes × 8 GPUs (64× H100 80GB total)

```yaml
# production-70b.yaml
job_id: "llama-70b-prod-v1"
model: "meta-llama/Llama-3.1-70B"
training_type: "qlora"

# Hardware (64 GPUs total)
nodes: 8
gpus_per_node: 8

# Storage (production S3 with lifecycle)
storage:
  type: "s3"
  use_instance_role: true  # IAM role (most secure)
  region: "us-east-1"
  dataset_uri: "s3://prod-training/datasets/enterprise-qa-v2.jsonl"
  output_uri: "s3://prod-models/llama-70b-enterprise-qa-v2"
  checkpoint_uri: "s3://prod-checkpoints/llama-70b-enterprise-qa-v2"

# Monitoring (all backends)
monitoring:
  wandb:
    project: "production-llama-70b"
    entity: "ml-production"
    tags: ["70b", "qlora", "enterprise", "v2"]
    wandb_log_model: "checkpoint"
  mlflow:
    tracking_uri: "https://mlflow.prod.company.com"
    experiment_name: "enterprise-qa-llama-70b"
    run_name: "v2-production-run"
    hf_mlflow_log_artifacts: true
  mongodb:
    uri: "mongodb+srv://mongo.prod.company.com"
    database: "training_prod"
    collection: "metrics"

# Hyperparameters (production-tuned)
hyperparameters:
  # Dataset
  sequence_len: 4096
  sample_packing: true
  pad_to_sequence_len: true
  dataset_num_proc: 32
  dataset_exact_deduplication: true

  # Batch size (calculated for 64 GPUs)
  micro_batch_size: 1
  gradient_accumulation_steps: 32
  # Effective batch = 1 × 32 × 64 = 2048

  # Learning
  learning_rate: 1e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  num_epochs: 2
  max_grad_norm: 1.0

  # LoRA (high capacity for complex task)
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  lora_target_linear: true
  lora_modules_to_save:
    - embed_tokens
    - lm_head

  # Optimizer
  optimizer: "paged_adamw_8bit"
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8

  # Precision
  bf16: true  # H100 supports bf16
  tf32: false
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  # Performance (H100 optimizations)
  flash_attention: true
  flash_attn_cross_entropy: true
  flash_attn_rms_norm: true
  chunked_cross_entropy: false

  # FSDP (required for 70B)
  fsdp_config:
    fsdp_sharding_strategy: "FULL_SHARD"
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
    fsdp_transformer_layer_cls_to_wrap: "LlamaDecoderLayer"
    fsdp_offload_params: false  # H100s have enough memory
    fsdp_sync_module_states: true
    fsdp_cpu_ram_efficient_loading: true
    fsdp_use_orig_params: false
    fsdp_limit_all_gathers: true
    fsdp_state_dict_type: "FULL_STATE_DICT"

  # Parallelism (optional for 70B)
  tensor_parallel_size: 2  # Split model across 2 GPUs

  # Evaluation (conservative for production)
  val_set_size: 0.02  # 2% (large dataset)
  evals_per_epoch: 4
  saves_per_epoch: 2
  save_total_limit: 5
  logging_steps: 10
  early_stopping_patience: 5

  # Monitoring
  include_tkps: true
  include_tokens_per_second: true

  # Safety
  loss_watchdog_threshold: 5.0
  loss_watchdog_patience: 3

  # Output
  hub_model_id: "company/llama-70b-enterprise-qa-v2"
  save_safetensors: true

output_dir: "/workspace/output"

# Lifecycle management (production features)
lifecycle:
  state_tracking:
    enabled: true
    mongodb:
      uri: "mongodb+srv://mongo.prod.company.com"
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
      to: ["ml-team@company.com", "oncall@company.com"]
      smtp:
        server: "smtp.company.com"
        port: 587
        use_tls: true
        secret_path: "smtp/credentials"

  server_deletion:
    enabled: true
    webhook:
      url: "https://api.company.com/training/complete"
      method: "POST"
      headers:
        Authorization: "Bearer ${API_TOKEN}"
    delete_on_success:
      enabled: true
      delay_minutes: 5
    delete_on_failure:
      enabled: true
      delay_hours: 1

# Secrets
secrets_manager:
  type: "vault"
  config:
    vault_addr: "https://vault.prod.company.com:8200"
    vault_token: "${VAULT_TOKEN}"
    mount_point: "secret"
```

---

### Example 6: Full Fine-Tuning (3B Model, 8 GPUs)

**Scenario**: Full parameter fine-tuning of smaller model

```yaml
# full-finetune-3b.yaml
job_id: "phi-3b-full-finetune"
model: "microsoft/Phi-3-mini-4k-instruct"
training_type: "full"  # Full fine-tuning

# Hardware
nodes: 1
gpus_per_node: 8

# Storage
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://bucket/specialized-domain.jsonl"
  output_uri: "s3://bucket/models/phi-3b-full"
  checkpoint_uri: "s3://bucket/checkpoints/phi-3b-full"

# Monitoring
monitoring:
  wandb:
    project: "full-finetuning"
    entity: "research-team"

# Hyperparameters
hyperparameters:
  # Dataset
  sequence_len: 2048
  sample_packing: true

  # Batch size (smaller for full fine-tuning)
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  # Effective batch = 1 × 16 × 8 = 128

  # Learning (lower LR for full fine-tuning)
  learning_rate: 5e-5
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01
  num_epochs: 3

  # Optimizer
  optimizer: "adamw_torch_fused"

  # Precision
  bf16: true
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  # Performance
  flash_attention: true

  # FSDP (for 3B model)
  fsdp_config:
    fsdp_sharding_strategy: "FULL_SHARD"
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
    fsdp_transformer_layer_cls_to_wrap: "PhiDecoderLayer"
    fsdp_sync_module_states: true
    fsdp_state_dict_type: "FULL_STATE_DICT"

  # Evaluation
  val_set_size: 0.05
  evals_per_epoch: 4
  saves_per_epoch: 2
  save_total_limit: 3
  logging_steps: 10

output_dir: "/workspace/output"
```

---

## Specialized Use Cases

### Example 7: Long Context Training (32k tokens)

**Scenario**: Training for long document processing

```yaml
# long-context-32k.yaml
job_id: "qwen-7b-longcontext-32k"
model: "Qwen/Qwen2.5-7B"
training_type: "qlora"

# Hardware (needs multiple GPUs for long context)
nodes: 2
gpus_per_node: 8

# Storage
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://bucket/long-documents.jsonl"
  output_uri: "s3://bucket/models/qwen-7b-32k"
  checkpoint_uri: "s3://bucket/checkpoints/qwen-7b-32k"

# Monitoring
monitoring:
  wandb:
    project: "long-context"

# Hyperparameters
hyperparameters:
  # Dataset (LONG context)
  sequence_len: 32768  # 32k tokens
  sample_packing: true
  pad_to_sequence_len: true

  # Batch size (tiny for long sequences)
  micro_batch_size: 1
  gradient_accumulation_steps: 64
  # Effective batch = 1 × 64 × 16 = 1024

  # Learning
  learning_rate: 1e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  num_epochs: 2

  # LoRA
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05

  # Optimizer
  optimizer: "paged_adamw_8bit"

  # Precision
  bf16: true
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false

  # Performance (critical for long context)
  flash_attention: true
  tiled_mlp: true  # ALST for memory-efficient long context
  tiled_mlp_num_shards: 4

  # Context parallelism (split sequences across GPUs)
  context_parallel_size: 4  # Split sequence 4 ways
  ring_attn_func: "varlen_llama3"

  # FSDP
  fsdp_config:
    fsdp_sharding_strategy: "FULL_SHARD"
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
    fsdp_transformer_layer_cls_to_wrap: "Qwen2DecoderLayer"
    fsdp_offload_params: false
    fsdp_sync_module_states: true
    fsdp_state_dict_type: "FULL_STATE_DICT"

  # Evaluation
  val_set_size: 0.02
  evals_per_epoch: 2
  saves_per_epoch: 2
  logging_steps: 10

output_dir: "/workspace/output"
```

---

### Example 8: Vision Model (Multimodal)

**Scenario**: Training vision-language model

```yaml
# vision-llama.yaml
job_id: "llava-7b-vision"
model: "llava-hf/llava-1.5-7b-hf"
training_type: "qlora"

# Hardware
nodes: 1
gpus_per_node: 4

# Storage
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://bucket/vision-qa.jsonl"  # With image_url content
  output_uri: "s3://bucket/models/llava-7b-custom"

# Monitoring
monitoring:
  wandb:
    project: "vision-finetuning"

# Hyperparameters
hyperparameters:
  # Dataset (multimodal)
  sequence_len: 2048
  sample_packing: false  # Don't pack multimodal

  # Batch size (smaller for vision)
  micro_batch_size: 1
  gradient_accumulation_steps: 32

  # Learning
  learning_rate: 2e-4
  num_epochs: 3

  # LoRA (target vision and text modules)
  lora_r: 32
  lora_alpha: 64
  lora_target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
    - mm_projector  # Vision projector

  # Multimodal settings
  image_size: 336

  # Optimizer
  optimizer: "paged_adamw_8bit"

  # Precision
  bf16: true
  gradient_checkpointing: true

  # Performance
  flash_attention: true

  # Evaluation
  val_set_size: 0.05
  evals_per_epoch: 2
  saves_per_epoch: 1

output_dir: "/workspace/output"
```

---

### Example 9: DPO/RLHF Training

**Scenario**: Reinforcement Learning from Human Feedback

```yaml
# dpo-training.yaml
job_id: "llama-8b-dpo"
model: "meta-llama/Llama-3.1-8B"
training_type: "lora"

# Hardware
nodes: 1
gpus_per_node: 4

# Storage
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://bucket/preference-data.jsonl"  # DPO format
  output_uri: "s3://bucket/models/llama-8b-dpo"

# Monitoring
monitoring:
  wandb:
    project: "rlhf-llama"

# Hyperparameters
hyperparameters:
  # RL Configuration
  rl: "dpo"  # Direct Preference Optimization
  rl_beta: 0.1
  dpo_label_smoothing: 0.0
  dpo_use_weighting: false

  # Dataset
  sequence_len: 2048
  sample_packing: false  # Don't pack for DPO
  max_prompt_len: 1024

  # Batch size
  micro_batch_size: 2
  gradient_accumulation_steps: 8

  # Learning
  learning_rate: 5e-6  # Lower for RL
  lr_scheduler: "linear"
  warmup_ratio: 0.1
  num_epochs: 1

  # LoRA
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  lora_target_linear: true

  # Optimizer
  optimizer: "adamw_8bit"

  # Precision
  bf16: true
  gradient_checkpointing: true

  # Evaluation
  val_set_size: 0.1
  evals_per_epoch: 4
  saves_per_epoch: 1

output_dir: "/workspace/output"
```

---

### Example 10: Continued Pretraining

**Scenario**: Domain adaptation via continued pretraining

```yaml
# continued-pretraining.yaml
job_id: "llama-8b-medical-pretrain"
model: "meta-llama/Llama-3.1-8B"
training_type: "full"

# Hardware
nodes: 4
gpus_per_node: 8

# Storage
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://bucket/medical-corpus-tokenized/"
  output_uri: "s3://bucket/models/llama-8b-medical"

# Monitoring
monitoring:
  wandb:
    project: "domain-adaptation"

# Hyperparameters
hyperparameters:
  # Pretraining dataset
  pretraining_dataset:
    - path: "s3://bucket/medical-corpus-tokenized/"
      type: "completion"

  # Dataset
  sequence_len: 4096
  pretraining_sample_concatenation: true
  streaming: true
  streaming_multipack_buffer_size: 50000

  # Batch size
  micro_batch_size: 2
  gradient_accumulation_steps: 16

  # Learning (lower LR for pretraining)
  learning_rate: 1e-5
  lr_scheduler: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.1
  num_epochs: 1
  max_steps: 100000

  # Optimizer
  optimizer: "adamw_torch_fused"

  # Precision
  bf16: true
  tf32: true
  gradient_checkpointing: true

  # Performance
  flash_attention: true

  # FSDP
  fsdp_config:
    fsdp_sharding_strategy: "FULL_SHARD"
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
    fsdp_transformer_layer_cls_to_wrap: "LlamaDecoderLayer"
    fsdp_sync_module_states: true
    fsdp_state_dict_type: "FULL_STATE_DICT"

  # Checkpointing (save frequently)
  save_steps: 1000
  save_total_limit: 10

  # No evaluation for pretraining
  val_set_size: 0.0

output_dir: "/workspace/output"
```

---

### Example 11: Memory-Constrained (70B on Limited GPUs)

**Scenario**: Training 70B on 4× A100 40GB (limited memory)

```yaml
# memory-constrained-70b.yaml
job_id: "llama-70b-memory-optimized"
model: "meta-llama/Llama-3.1-70B"
training_type: "qlora"

# Hardware (only 4 GPUs)
nodes: 1
gpus_per_node: 4

# Storage
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://bucket/dataset.jsonl"
  output_uri: "s3://bucket/models/llama-70b-output"

# Auto-config (memory mode)
auto_config:
  enabled: true
  optimize_for: "memory"  # Aggressive memory saving
  sequence_length: 2048
  sample_packing: true

# Manual overrides
hyperparameters:
  # Force minimum batch size
  micro_batch_size: 1
  gradient_accumulation_steps: 64

  # Learning
  learning_rate: 1e-4
  num_epochs: 2

  # LoRA
  lora_r: 32
  lora_alpha: 64

  # Memory optimizations
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  activation_offloading: true  # Offload activations

  # FSDP with CPU offloading
  fsdp_config:
    fsdp_sharding_strategy: "FULL_SHARD"
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
    fsdp_transformer_layer_cls_to_wrap: "LlamaDecoderLayer"
    fsdp_offload_params: true  # CPU offloading
    fsdp_cpu_ram_efficient_loading: true
    fsdp_sync_module_states: true
    fsdp_state_dict_type: "FULL_STATE_DICT"

  # Evaluation (minimal)
  val_set_size: 0.02
  evals_per_epoch: 1
  saves_per_epoch: 1

output_dir: "/workspace/output"
```

---

### Example 12: Maximum Speed (7B, 8 GPUs)

**Scenario**: Optimize for fastest training time

```yaml
# speed-optimized-7b.yaml
job_id: "llama-7b-speed"
model: "meta-llama/Llama-3.1-8B"
training_type: "lora"  # LoRA faster than QLoRA

# Hardware
nodes: 1
gpus_per_node: 8

# Storage
storage:
  type: "filesystem"  # Local for speed
  dataset_uri: "/nvme/datasets/data.jsonl"
  output_uri: "/nvme/output"

# Auto-config (speed mode)
auto_config:
  enabled: true
  optimize_for: "speed"
  sequence_length: 2048
  sample_packing: true

# Hyperparameters
hyperparameters:
  # Maximize batch size
  micro_batch_size: 8  # Push to limit
  gradient_accumulation_steps: 2  # Minimal

  # Learning
  learning_rate: 3e-4
  num_epochs: 1  # Quick training

  # LoRA (lower rank for speed)
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.0  # No dropout for speed

  # Optimizer (fastest)
  optimizer: "adamw_torch_fused"

  # Precision
  bf16: true
  tf32: true  # Extra speed on Ampere
  gradient_checkpointing: false  # Disabled for speed

  # Performance (all optimizations)
  flash_attention: true
  lora_mlp_kernel: true
  lora_qkv_kernel: true
  lora_o_kernel: true
  torch_compile: true
  torch_compile_mode: "max-autotune"

  # Dataloader (maximize throughput)
  dataloader_num_workers: 8
  dataloader_pin_memory: true
  dataloader_prefetch_factor: 512

  # Minimal evaluation
  val_set_size: 0.01
  evals_per_epoch: 1
  saves_per_epoch: 1
  logging_steps: 50

output_dir: "/nvme/output"
```

---

## Configuration Patterns

### Pattern 1: Development → Staging → Production

**Development**:
```yaml
job_id: "dev-test"
hyperparameters:
  micro_batch_size: 1
  max_steps: 100  # Quick test
  val_set_size: 0.2
```

**Staging**:
```yaml
job_id: "staging-validation"
hyperparameters:
  micro_batch_size: 2
  num_epochs: 1
  val_set_size: 0.1
```

**Production**:
```yaml
job_id: "prod-v1"
hyperparameters:
  micro_batch_size: 4
  num_epochs: 3
  val_set_size: 0.05
  early_stopping_patience: 5
```

### Pattern 2: Iterative Hyperparameter Tuning

```yaml
# Run 1: Baseline
job_id: "tune-run-1-baseline"
hyperparameters:
  learning_rate: 2e-4
  lora_r: 32

# Run 2: Higher LR
job_id: "tune-run-2-higher-lr"
hyperparameters:
  learning_rate: 5e-4
  lora_r: 32

# Run 3: Larger LoRA
job_id: "tune-run-3-larger-lora"
hyperparameters:
  learning_rate: 2e-4
  lora_r: 64
```

---

## Next Steps

- **[Job Configuration](JOB_CONFIG.md)** - Complete config reference
- **[Hyperparameters Guide](HYPERPARAMETERS.md)** - Parameter tuning
- **[Dataset Format](DATASET_FORMAT.md)** - Data preparation
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
