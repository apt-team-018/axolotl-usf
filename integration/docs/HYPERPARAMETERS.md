# Hyperparameters Guide

Complete guide to training hyperparameters with ideal values for different scenarios.

## Table of Contents
- [Overview](#overview)
- [Batch Size Configuration](#batch-size-configuration)
- [Learning Rate](#learning-rate)
- [LoRA Parameters](#lora-parameters)
- [Optimizer Selection](#optimizer-selection)
- [Learning Rate Schedulers](#learning-rate-schedulers)
- [Sequence Length](#sequence-length)
- [Training Duration](#training-duration)
- [Evaluation Settings](#evaluation-settings)
- [Recommended Configurations](#recommended-configurations)
- [Tuning Guide](#tuning-guide)

## Overview

Hyperparameters control how your model trains. The right settings depend on:
- **Model size** (1B, 7B, 70B, etc.)
- **Training type** (full, LoRA, QLoRA)
- **Hardware** (GPU type, memory, count)
- **Dataset** (size, quality, task)
- **Goals** (speed vs quality)

### Default Behavior

If you don't specify hyperparameters, the system uses sensible defaults:

```yaml
# These are applied automatically if not specified
hyperparameters:
  micro_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  num_epochs: 3
  sequence_length: 2048
  optimizer: "paged_adamw_8bit"
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
```

---

## Batch Size Configuration

The most important hyperparameters for memory and speed.

### Understanding Batch Sizes

**Three types of batch size**:

1. **Micro Batch Size** - Batch size per GPU per forward pass
2. **Gradient Accumulation Steps** - How many micro batches before optimizer step
3. **Effective Batch Size** - Total samples per optimization step

**Formula**:
```
Effective Batch Size = micro_batch_size × gradient_accumulation_steps × num_gpus
```

### Example Calculation

```yaml
# Configuration
micro_batch_size: 2
gradient_accumulation_steps: 16
# Hardware: 4 GPUs

# Effective batch size = 2 × 16 × 4 = 128
```

### micro_batch_size

```yaml
micro_batch_size: 2  # Samples per GPU per step
```

**Range**: 1-64 (typically 1-8 for large models)

**How to choose**:
1. Start with 1
2. Increase until GPU memory is 90-95% full
3. Monitor with `nvidia-smi`

**By Model Size**:
- **1B-3B models**: 4-8
- **7B-13B models**: 2-4
- **30B-70B models**: 1-2
- **405B+ models**: 1

**By Training Type**:
- **Full fine-tuning**: Lower (1-2)
- **LoRA**: Medium (2-4)
- **QLoRA**: Higher (2-8)

### gradient_accumulation_steps

```yaml
gradient_accumulation_steps: 16
```

**Range**: 1-128

**Purpose**: Simulate larger batch sizes without more memory

**How to choose**:
```python
# Target effective batch size: 128
# GPUs: 4
# Max micro_batch_size: 2 (limited by memory)

gradient_accumulation_steps = 128 / (2 × 4) = 16
```

**Trade-off**:
- ✅ Larger steps = more stable training
- ❌ Larger steps = slower (more forward passes per update)

### Effective Batch Size Guidelines

**Recommended ranges**:
- **Small models (1B-3B)**: 32-128
- **Medium models (7B-13B)**: 64-256
- **Large models (30B-70B)**: 128-512
- **Very large (405B+)**: 256-1024

**Quality vs Speed**:
- **Quality priority**: 128-256 (more stable)
- **Speed priority**: 32-64 (faster iterations)
- **Balanced**: 64-128

---

## Learning Rate

The most critical hyperparameter for training success.

### Basics

```yaml
learning_rate: 2e-4  # 0.0002
```

**Too high**: Training diverges, loss explodes
**Too low**: Training too slow, may not converge
**Just right**: Steady loss decrease

### Recommended Values

**By Training Type**:

```yaml
# Full Fine-Tuning
learning_rate: 1e-5  # to 5e-5

# LoRA
learning_rate: 1e-4  # to 3e-4

# QLoRA
learning_rate: 2e-4  # to 5e-4
```

**By Model Size**:

| Model Size | Full | LoRA | QLoRA |
|------------|------|------|-------|
| 1B-3B | 5e-5 | 3e-4 | 5e-4 |
| 7B-13B | 2e-5 | 2e-4 | 3e-4 |
| 30B-70B | 1e-5 | 1e-4 | 2e-4 |
| 405B+ | 5e-6 | 5e-5 | 1e-4 |

**General Rule**: Smaller models can use higher learning rates.

### Learning Rate Warmup

```yaml
warmup_ratio: 0.1  # 10% of training
```

**What it does**: Gradually increases LR from 0 to target over first N% of training

**Recommended values**:
- **Short training (<1 epoch)**: 0.03-0.05
- **Normal training (2-3 epochs)**: 0.1
- **Long training (>5 epochs)**: 0.05-0.1

**Why use it**: Prevents early instability, especially with large learning rates

### Learning Rate Schedule

See [Learning Rate Schedulers](#learning-rate-schedulers) section.

---

## LoRA Parameters

For `lora` and `qlora` training types only.

### lora_r (Rank)

```yaml
lora_r: 32
```

**Range**: 4-256 (powers of 2 recommended)

**What it controls**: Capacity of LoRA adapter

**Common values**:
- `r=8`: Minimal adaptation (simple tasks)
- `r=16`: Good for most tasks
- `r=32`: Higher capacity (complex tasks) **[RECOMMENDED]**
- `r=64`: Maximum capacity (very complex tasks)

**Trade-offs**:
- Higher `r` = more parameters = more memory + slower
- Higher `r` = more capacity = better for complex tasks

**Guidelines**:
- **Simple tasks** (sentiment, classification): 8-16
- **Moderate tasks** (QA, summarization): 16-32
- **Complex tasks** (reasoning, multi-task): 32-64
- **Very complex** (domain adaptation): 64-128

### lora_alpha

```yaml
lora_alpha: 64  # Usually 2× lora_r
```

**Relationship to rank**:
```yaml
# Typical configuration
lora_r: 32
lora_alpha: 64  # = 2 × r
```

**What it controls**: Scaling factor for LoRA updates

**Rule of thumb**: Set to `2 × lora_r`

**Alternative strategies**:
- `alpha = r`: More conservative updates
- `alpha = 2×r`: Standard (recommended)
- `alpha = 4×r`: Aggressive updates

### lora_dropout

```yaml
lora_dropout: 0.05  # 5% dropout
```

**Range**: 0.0-0.2

**Recommended values**:
- **Small datasets (<10k samples)**: 0.05-0.1
- **Medium datasets (10k-100k)**: 0.05
- **Large datasets (>100k)**: 0.0-0.05

**Purpose**: Regularization to prevent overfitting

### lora_target_modules

```yaml
# Auto-detected based on model, but you can override
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

**Standard configs**:
- **Attention only**: `[q_proj, k_proj, v_proj, o_proj]`
- **Attention + MLP**: All modules (recommended)
- **MLP only**: `[gate_proj, up_proj, down_proj]`

---

## Optimizer Selection

```yaml
optimizer: "paged_adamw_8bit"
```

### Available Optimizers

| Optimizer | Memory | Speed | Quality | Best For |
|-----------|--------|-------|---------|----------|
| `paged_adamw_8bit` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | QLoRA, limited memory |
| `adamw_8bit` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Memory-constrained |
| `adamw_torch_fused` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Speed priority, full precision |
| `adamw_torch` | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Standard |
| `sgd` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Rarely used |

### Recommendations

```yaml
# QLoRA (memory priority)
optimizer: "paged_adamw_8bit"

# LoRA (balanced)
optimizer: "adamw_8bit"

# Full fine-tuning (quality priority)
optimizer: "adamw_torch_fused"
```

### Optimizer Hyperparameters

```yaml
optimizer: "adamw_torch"
weight_decay: 0.01      # L2 regularization
adam_beta1: 0.9         # Momentum (rarely changed)
adam_beta2: 0.999       # RMSprop momentum (rarely changed)
adam_epsilon: 1e-8      # Numerical stability (rarely changed)
```

**weight_decay**:
- **Default**: 0.01
- **No regularization**: 0.0
- **Strong regularization**: 0.1

---

## Learning Rate Schedulers

```yaml
lr_scheduler: "cosine"
```

### Available Schedulers

**cosine** (RECOMMENDED):
```yaml
lr_scheduler: "cosine"
warmup_ratio: 0.1
```
- Smooth decay from max to min
- Best for most use cases
- Widely used in research

**linear**:
```yaml
lr_scheduler: "linear"
warmup_ratio: 0.1
```
- Linear decay from max to 0
- Good for short training runs

**constant**:
```yaml
lr_scheduler: "constant"
warmup_ratio: 0.1
```
- LR stays constant after warmup
- Use for continued training

**constant_with_warmup**:
```yaml
lr_scheduler: "constant_with_warmup"
warmup_ratio: 0.1
```
- Warmup then constant
- Good for debugging

**polynomial**:
```yaml
lr_scheduler: "polynomial"
lr_scheduler_power: 1.0
```
- Polynomial decay
- Flexible decay rate

### Visualization

```
LR
│
│  ╱────╲              cosine (recommended)
│ ╱      ────╲
│╱           ───╲
└─────────────────── Steps

│
│  ╱────────╲          linear
│ ╱          ╲
│╱            ╲
└─────────────────── Steps

│
│  ╱──────────        constant_with_warmup
│ ╱
│╱
└─────────────────── Steps
```

---

## Sequence Length

```yaml
sequence_length: 2048
```

**What it controls**: Maximum input/output length

### Choosing Sequence Length

**By task type**:
- **Short text** (classification, NER): 512-1024
- **Medium text** (QA, summarization): 2048-4096
- **Long text** (documents, articles): 4096-8192
- **Very long** (books, reports): 8192-32768

**By model**:
Check model's native `max_position_embeddings`:
- Llama 3.1: 131,072 (128k)
- Mistral: 32,768 (32k)
- Qwen 2.5: 131,072 (128k)

**Memory impact**:
```
Memory ∝ sequence_length²

2048 → 4096 = 4× more memory
4096 → 8192 = 4× more memory
```

**Recommendations**:
1. Use shortest length that fits your data
2. Check dataset: `max(len(sample) for sample in dataset)`
3. Add 10-20% buffer for safety
4. Round to powers of 2: 512, 1024, 2048, 4096, 8192

### sample_packing

```yaml
sample_packing: true  # Recommended
pad_to_sequence_len: true
```

**What it does**: Packs multiple short samples into one sequence

**Benefits**:
- ✅ Better GPU utilization
- ✅ Faster training
- ✅ No wasted computation on padding

**When to use**:
- ✅ Variable-length samples
- ✅ Many short samples
- ❌ All samples near max length

---

## Training Duration

```yaml
num_epochs: 3
```

### num_epochs

**How many passes through the dataset**

**Guidelines**:
- **Large dataset (>100k samples)**: 1-2 epochs
- **Medium dataset (10k-100k)**: 2-3 epochs
- **Small dataset (<10k)**: 3-5 epochs
- **Tiny dataset (<1k)**: 5-10 epochs (risk of overfitting)

**Signs of overfitting**:
- Training loss decreases, validation loss increases
- Perfect training accuracy, poor validation
- Model memorizes training data

**Early stopping** (recommended):
```yaml
early_stopping_patience: 3  # Stop if no improvement for 3 evals
```

### max_steps

```yaml
# Alternative to num_epochs
max_steps: 10000  # Train for exactly 10k steps
```

**Use when**:
- You know exact step count needed
- Comparing different configurations
- Reproducing research results

**Relationship**:
```
total_steps = (dataset_size / effective_batch_size) × num_epochs
```

---

## Evaluation Settings

```yaml
val_set_size: 0.05        # 5% for validation
evals_per_epoch: 2        # Evaluate 2× per epoch
saves_per_epoch: 1        # Save checkpoint 1× per epoch
logging_steps: 10         # Log metrics every 10 steps
```

### val_set_size

**Validation split percentage**

```yaml
val_set_size: 0.05  # 5% of dataset
```

**Guidelines**:
- **Large dataset (>100k)**: 0.01-0.05 (1-5%)
- **Medium (10k-100k)**: 0.05-0.1 (5-10%)
- **Small (<10k)**: 0.1-0.2 (10-20%)

**No validation**:
```yaml
val_set_size: 0.0  # Use all data for training
```

### evals_per_epoch

```yaml
evals_per_epoch: 2  # Evaluate 2× per epoch
```

**Balance**: More evals = better monitoring, but slower training

**Recommendations**:
- **Short training (<3 epochs)**: 4-8 evals/epoch
- **Normal training (3-5 epochs)**: 2-4 evals/epoch
- **Long training (>5 epochs)**: 1-2 evals/epoch

### saves_per_epoch

```yaml
saves_per_epoch: 1  # Save checkpoint 1× per epoch
```

**Disk space consideration**: More saves = more disk usage

**Recommendations**:
- **Production**: 1-2 saves/epoch
- **Experiments**: 1 save/epoch
- **Long runs**: 2-4 saves/epoch (in case of failures)

**Automatic cleanup**: System keeps only last N checkpoints (configurable)

### logging_steps

```yaml
logging_steps: 10  # Log every 10 optimization steps
```

**Balance**: More logging = better visibility, but more overhead

**Recommendations**:
- **Short runs**: 1-5 steps
- **Normal runs**: 10-50 steps
- **Very long runs**: 100+ steps

---

## Recommended Configurations

### 1B-3B Model (Single GPU)

```yaml
hyperparameters:
  # Batch size
  micro_batch_size: 4
  gradient_accumulation_steps: 8
  # LR: 2-4 × 32 = 128

  # Learning
  learning_rate: 3e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01

  # Duration
  num_epochs: 3

  # Sequence
  sequence_length: 2048

  # LoRA
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05

  # Optimizer
  optimizer: "adamw_8bit"

  # Eval
  val_set_size: 0.05
  evals_per_epoch: 4
  saves_per_epoch: 1
  logging_steps: 10
```

### 7B Model (4 GPUs, QLoRA)

```yaml
hyperparameters:
  # Batch size
  micro_batch_size: 2
  gradient_accumulation_steps: 16
  # EBS: 2 × 16 × 4 = 128

  # Learning
  learning_rate: 2e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01

  # Duration
  num_epochs: 3

  # Sequence
  sequence_length: 2048
  sample_packing: true

  # LoRA
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05

  # Optimizer
  optimizer: "paged_adamw_8bit"

  # Eval
  val_set_size: 0.05
  evals_per_epoch: 2
  saves_per_epoch: 1
  logging_steps: 10
```

### 70B Model (32 GPUs, QLoRA + FSDP)

```yaml
hyperparameters:
  # Batch size
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  # EBS: 1 × 16 × 32 = 512

  # Learning
  learning_rate: 1e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.1
  weight_decay: 0.01

  # Duration
  num_epochs: 2

  # Sequence
  sequence_length: 4096
  sample_packing: true

  # LoRA
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.05

  # Optimizer
  optimizer: "paged_adamw_8bit"

  # Advanced
  fsdp_offload: false  # H100s have enough memory

  # Eval
  val_set_size: 0.02
  evals_per_epoch: 4
  saves_per_epoch: 2
  logging_steps: 10
```

---

## Tuning Guide

### Step-by-Step Tuning Process

**1. Start with defaults**:
```yaml
# Use auto-config or defaults
auto_config:
  enabled: true
  optimize_for: "balanced"
```

**2. Maximize micro_batch_size**:
```bash
# Increase until OOM, then reduce by 1
micro_batch_size: 1  # Start
micro_batch_size: 2  # OK
micro_batch_size: 4  # OK
micro_batch_size: 8  # OOM! → Use 4
```

**3. Adjust gradient accumulation**:
```yaml
# To hit target effective batch size
target_batch = 128
micro_batch = 4
gpus = 4
grad_accum = 128 / (4 × 4) = 8
```

**4. Tune learning rate**:
```yaml
# Run short test with different LRs
learning_rate: [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
# Pick the one with fastest loss decrease
```

**5. Adjust LoRA rank** (if needed):
```yaml
# Start with 32
# If underfitting: increase to 64
# If overfitting: decrease to 16
```

### Monitoring Training

**Watch these metrics**:
1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease (if not, overfitting)
3. **Learning Rate**: Should follow schedule
4. **GPU Memory**: Should be 90-95% utilized
5. **Throughput**: Tokens/second

**Red flags**:
- 🚩 Loss = NaN (learning rate too high)
- 🚩 Loss increasing (bad hyperparameters)
- 🚩 No improvement after many steps (learning rate too low)
- 🚩 Train/val loss diverging (overfitting)

---

## Next Steps

- **[Auto-Optimizer](AUTO_OPTIMIZER.md)** - Automatic hyperparameter tuning
- **[Performance Tuning](PERFORMANCE.md)** - Speed and memory optimization
- **[Examples](EXAMPLES.md)** - Complete configuration examples
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
