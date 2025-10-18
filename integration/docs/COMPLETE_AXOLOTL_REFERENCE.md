
# Complete Axolotl Configuration Reference

Comprehensive reference for ALL Axolotl configuration parameters (180+ parameters).

## Table of Contents
- [How to Use with Integration](#how-to-use-with-integration)
- [Model Configuration](#model-configuration)
- [Dataset Configuration](#dataset-configuration)
- [Training Hyperparameters](#training-hyperparameters)
- [LoRA/PEFT Configuration](#lorapeft-configuration)
- [Sequence & Packing](#sequence--packing)
- [Precision & Mixed Precision](#precision--mixed-precision)
- [Performance Optimizations](#performance-optimizations)
- [FSDP Configuration](#fsdp-configuration)
- [DeepSpeed Configuration](#deepspeed-configuration)
- [Distributed & Parallelism](#distributed--parallelism)
- [Evaluation & Checkpointing](#evaluation--checkpointing)
- [Monitoring & Logging](#monitoring--logging)
- [Advanced Features](#advanced-features)
- [RL Training](#rl-training)
- [Multimodal](#multimodal)

---

## How to Use with Integration

**ALL Axolotl parameters** can be used through the integration's `hyperparameters` section:

```yaml
# job_config.yaml
job_id: "my-job"
model: "meta-llama/Llama-3.1-8B"
training_type: "qlora"
nodes: 1
gpus_per_node: 4

storage:
  type: "s3"
  dataset_uri: "s3://bucket/data.jsonl"

# ANY Axolotl parameter goes here:
hyperparameters:
  sequence_len: 2048
  sample_packing: true
  flash_attention: true
  # ... ALL 180+ params from this reference
```

---

## Model Configuration

Parameters for model loading and initialization.

### base_model
**Type**: `string`
**Required**: ✅ Yes
**Default**: None

HuggingFace model ID or local path.

```yaml
base_model: "meta-llama/Llama-3.1-8B"
# Or local:
base_model: "/models/my-model"
```

### model_type
**Type**: `string`
**Required**: No
**Default**: Auto-detected

Model class name for loading.

```yaml
model_type: "LlamaForCausalLM"
model_type: "MistralForCausalLM"
model_type: "Qwen2ForCausalLM"
```

**When to use**: Usually auto-detected, only needed if detection fails.

### tokenizer_type
**Type**: `string`
**Required**: No
**Default**: Auto-detected

Tokenizer class name.

```yaml
tokenizer_type: "LlamaTokenizer"
tokenizer_type: "AutoTokenizer"
```

### base_model_config
**Type**: `string`
**Required**: No
**Default**: Same as `base_model`

Path to model config if different from base_model.

```yaml
base_model_config: "meta-llama/Llama-3.1-8B"
```

### tokenizer_config
**Type**: `string`
**Required**: No
**Default**: Same as `base_model`

Path to tokenizer config.

```yaml
tokenizer_config: "meta-llama/Llama-3.1-8B"
```

### processor_type
**Type**: `string`
**Required**: No (for multimodal models)
**Default**: Auto-detected

Processor class for multimodal models.

```yaml
processor_type: "LlavaProcessor"
```

### trust_remote_code
**Type**: `boolean`
**Required**: No
**Default**: `false`

Allow executing remote code when loading model.

```yaml
trust_remote_code: true
```

**⚠️ Security Warning**: Only use with trusted sources.

### tokenizer_use_fast
**Type**: `boolean`
**Required**: No
**Default**: `true`

Use fast tokenizer (Rust-based).

```yaml
tokenizer_use_fast: true
```

### tokenizer_legacy
**Type**: `boolean`
**Required**: No
**Default**: `true`

Use legacy tokenizer behavior.

```yaml
tokenizer_legacy: false
```

### tokenizer_use_mistral_common
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use mistral-common tokenizer for Mistral models.

```yaml
tokenizer_use_mistral_common: true
```

### load_in_8bit
**Type**: `boolean`
**Required**: No
**Default**: `false`

Load model in 8-bit quantization.

```yaml
load_in_8bit: true
```

**Use with**: LoRA training (not QLoRA)

### load_in_4bit
**Type**: `boolean`
**Required**: No
**Default**: `false`

Load model in 4-bit quantization (QLoRA).

```yaml
load_in_4bit: true
```

**Required for**: QLoRA training

### gptq
**Type**: `boolean`
**Required**: No
**Default**: `false`

Whether model is GPTQ quantized.

```yaml
gptq: true
```

### bnb_config_kwargs
**Type**: `dict`
**Required**: No
**Default**: None

Override bitsandbytes quantization config.

```yaml
bnb_config_kwargs:
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
```

### hub_model_id
**Type**: `string`
**Required**: No
**Default**: None

Push checkpoints to this HuggingFace repo.

```yaml
hub_model_id: "username/my-finetuned-model"
```

### hub_strategy
**Type**: `string`
**Required**: No
**Default**: `"every_save"`

How to push to hub.

```yaml
hub_strategy: "every_save"  # or "end", "checkpoint"
```

### save_safetensors
**Type**: `boolean`
**Required**: No
**Default**: `true`

Save model in safetensors format.

```yaml
save_safetensors: true
```

### resize_token_embeddings_to_32x
**Type**: `boolean`
**Required**: No
**Default**: `false`

Resize embeddings to multiples of 32 (reported to improve speed).

```yaml
resize_token_embeddings_to_32x: true
```

### shrink_embeddings
**Type**: `boolean`
**Required**: No
**Default**: `false`

Shrink embeddings if tokenizer vocab is smaller.

```yaml
shrink_embeddings: true
```

### embeddings_skip_upcast
**Type**: `boolean`
**Required**: No
**Default**: `false`

Don't upcast embeddings to float32 with PEFT (saves memory).

```yaml
embeddings_skip_upcast: true
```

### reinit_weights
**Type**: `boolean`
**Required**: No
**Default**: `false`

Randomly reinitialize weights instead of loading pretrained.

```yaml
reinit_weights: true
```

**Use case**: Training from scratch (rare)

### low_cpu_mem_usage
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use less CPU memory when loading model.

```yaml
low_cpu_mem_usage: true
```

### device_map
**Type**: `string` or `dict`
**Required**: No
**Default**: `"auto"`

How to distribute model across devices.

```yaml
device_map: "auto"
device_map: "sequential"
device_map: {"": 0}  # All on GPU 0
```

### max_memory
**Type**: `dict`
**Required**: No
**Default**: None

Max memory per device.

```yaml
max_memory:
  0: "40GiB"  # GPU 0
  1: "40GiB"  # GPU 1
  cpu: "100GiB"
```

### gpu_memory_limit
**Type**: `int` or `string`
**Required**: No
**Default**: None

Limit GPU memory (in GB).

```yaml
gpu_memory_limit: 40  # 40GB
gpu_memory_limit: "40GiB"
```

### model_quantization_config
**Type**: `string`
**Required**: No
**Default**: None

Model loading quantization config.

```yaml
model_quantization_config: "Mxfp4Config"
```

### model_quantization_config_kwargs
**Type**: `dict`
**Required**: No
**Default**: None

Kwargs for quantization config.

```yaml
model_quantization_config_kwargs:
  block_size: 128
```

---

## Dataset Configuration

Parameters for dataset loading and processing.

### datasets
**Type**: `list`
**Required**: ✅ Yes
**Default**: None

List of datasets to train on.

```yaml
datasets:
  - path: "/data/dataset.jsonl"
    type: "chat_template"
```

**Required fields per dataset**:
- `path`: Dataset path or HF repo
- `type`: Format type (use `chat_template` for OpenAI format)

**Optional fields**:
- `chat_template`: Template name
- `split`: Dataset split to use
- `weight`: Sampling weight
- `shards`: Number of shards

**Multiple datasets**:
```yaml
datasets:
  - path: "/data/dataset1.jsonl"
    type: "chat_template"
    weight: 0.7
  - path: "/data/dataset2.jsonl"
    type: "chat_template"
    weight: 0.3
```

### test_datasets
**Type**: `list`
**Required**: No
**Default**: None

Datasets for evaluation.

```yaml
test_datasets:
  - path: "/data/test.jsonl"
    type: "chat_template"
```

**Note**: Use either `test_datasets` OR `val_set_size`, not both.

### dataset_type
**Type**: `string`
**Required**: No
**Default**: `"chat_template"` (enforced by integration)

**Integration enforces**: `"chat_template"` for OpenAI format.

### chat_template
**Type**: `string`
**Required**: No
**Default**: Auto-detected from model

Chat template to use.

```yaml
chat_template: "qwen3"
chat_template: "llama3"
chat_template: "tokenizer_default"
```

**Available templates**:
- `tokenizer_default` - Use model's template
- `llama3` - Llama 3.x
- `qwen3` - Qwen 2.5+
- `mistral` - Mistral
- `chatml` - ChatML
- `phi_3` - Phi-3
- `gemma` - Gemma
- `deepseek_v2` - DeepSeek V2
- `jinja` - Custom jinja

**Auto-detection**: If not specified, system detects from model name.

### chat_template_jinja
**Type**: `string`
**Required**: No (only if `chat_template: "jinja"`)
**Default**: None

Custom jinja template or path.

```yaml
chat_template: "jinja"
chat_template_jinja: |
  {% for message in messages %}
  <|{{ message.role }}|>{{ message.content }}<|end|>
  {% endfor %}
```

### chat_template_kwargs
**Type**: `dict`
**Required**: No
**Default**: None

Additional kwargs for chat template.

```yaml
chat_template_kwargs:
  thinking: false
  add_generation_prompt: true
```

### dataset_prepared_path
**Type**: `string`
**Required**: No
**Default**: None

Cache processed dataset here.

```yaml
dataset_prepared_path: "last_run_prepared"
dataset_prepared_path: "/cache/processed_data"
```

**Benefit**: Faster subsequent runs (skip preprocessing).

### dataset_num_proc
**Type**: `int`
**Required**: No
**Default**: `os.cpu_count()`

Parallel processes for dataset processing.

```yaml
dataset_num_proc: 8
```

### dataset_shard_num
**Type**: `int`
**Required**: No
**Default**: None

Total number of dataset shards.

```yaml
dataset_shard_num: 10
```

### dataset_shard_idx
**Type**: `int`
**Required**: No
**Default**: None

Which shard to use (0-indexed).

```yaml
dataset_shard_idx: 0  # Use first shard
```

### skip_prepare_dataset
**Type**: `boolean`
**Required**: No
**Default**: `false`

Skip dataset preparation (use cached).

```yaml
skip_prepare_dataset: true
```

### num_dataset_shards_to_save
**Type**: `int`
**Required**: No
**Default**: None

Number of shards to save prepared dataset.

```yaml
num_dataset_shards_to_save: 10
```

### shuffle_merged_datasets
**Type**: `boolean`
**Required**: No
**Default**: `true`

Shuffle after merging datasets.

```yaml
shuffle_merged_datasets: true
```

### shuffle_before_merging_datasets
**Type**: `boolean`
**Required**: No
**Default**: `false`

Shuffle each dataset before merging (for curriculum learning).

```yaml
shuffle_before_merging_datasets: true
```

### dataset_exact_deduplication
**Type**: `boolean`
**Required**: No
**Default**: `false`

Remove exact duplicate entries.

```yaml
dataset_exact_deduplication: true
```

### dataset_keep_in_memory
**Type**: `boolean`
**Required**: No
**Default**: `false`

Keep dataset in RAM during preprocessing.

```yaml
dataset_keep_in_memory: true
```

### push_dataset_to_hub
**Type**: `string`
**Required**: No
**Default**: None

Push prepared dataset to HuggingFace Hub.

```yaml
push_dataset_to_hub: "username/prepared-dataset"
```

### hf_use_auth_token
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use HF auth token for private datasets.

```yaml
hf_use_auth_token: true
```

### pretraining_dataset
**Type**: `list`
**Required**: No
**Default**: None

Dataset for pretraining (completion-style).

```yaml
pretraining_dataset:
  - path: "HuggingFaceFW/fineweb-edu"
    split: "train"
```

### pretraining_sample_concatenation
**Type**: `boolean`
**Required**: No
**Default**: `false`

Concatenate samples for pretraining.

```yaml
pretraining_sample_concatenation: true
```

### streaming
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use streaming mode (don't download entire dataset).

```yaml
streaming: true
```

### streaming_multipack_buffer_size
**Type**: `int`
**Required**: No
**Default**: `10000`

Buffer size for streaming multipack.

```yaml
streaming_multipack_buffer_size: 50000
```

---

## Training Hyperparameters

Core training parameters.

### micro_batch_size
**Type**: `int`
**Required**: No
**Default**: `1`

Samples per GPU per forward pass.

```yaml
micro_batch_size: 2
```

**How to choose**: Increase until GPU memory is 90-95% full.

### gradient_accumulation_steps
**Type**: `int`
**Required**: No
**Default**: `1`

Accumulate gradients for N steps before optimizer update.

```yaml
gradient_accumulation_steps: 16
```

**Effective batch size** = `micro_batch_size × gradient_accumulation_steps × num_gpus`

### batch_size
**Type**: `int`
**Required**: No
**Default**: Auto-calculated

Total batch size (not recommended to set manually).

```yaml
# Let system calculate:
# batch_size = micro_batch_size × gradient_accumulation_steps
```

### eval_batch_size
**Type**: `int`
**Required**: No
**Default**: Same as `micro_batch_size`

Batch size for evaluation.

```yaml
eval_batch_size: 4
```

### auto_find_batch_size
**Type**: `boolean`
**Required**: No
**Default**: `false`

Automatically find batch size that fits in memory.

```yaml
auto_find_batch_size: true
```

### learning_rate
**Type**: `float`
**Required**: ✅ Yes
**Default**: None

Learning rate for optimizer.

```yaml
learning_rate: 2e-4  # 0.0002
```

**Typical values**:
- Full fine-tuning: `1e-5` to `5e-5`
- LoRA: `1e-4` to `3e-4`
- QLoRA: `2e-4` to `5e-4`

### optimizer
**Type**: `string`
**Required**: No
**Default**: `"adamw_torch_fused"`

Optimizer to use.

```yaml
optimizer: "paged_adamw_8bit"
```

**Options**:
- `adamw_torch_fused` - Fastest, full precision (default)
- `paged_adamw_8bit` - Memory-efficient (recommended for QLoRA)
- `adamw_8bit` - 8-bit AdamW
- `adamw_bnb_8bit` - bitsandbytes 8-bit
- `adamw_torch` - Standard AdamW
- `adamw_anyprecision` - Custom precision
- `adafactor` - Adafactor
- `sgd` - SGD
- `came` - CAME optimizer
- `adopt_adamw` - ADOPT (torch >=2.5.1)
- `galore_adamw` - GaLore
- `galore_adamw_8bit` - GaLore 8-bit

### optim_args
**Type**: `string` or `dict`
**Required**: No
**Default**: None

Additional optimizer arguments.

```yaml
optim_args: "beta1=0.9,beta2=0.95"
# Or:
optim_args:
  beta1: 0.9
  beta2: 0.95
```

### optim_target_modules
**Type**: `list` or `string`
**Required**: No
**Default**: None

Target modules for optimization (GaLore).

```yaml
optim_target_modules: "all_linear"
# Or specific:
optim_target_modules:
  - "q_proj"
  - "v_proj"
```

### lr_scheduler
**Type**: `string`
**Required**: No
**Default**: `"cosine"`

Learning rate schedule.

```yaml
lr_scheduler: "cosine"
```

**Options**:
- `cosine` - Cosine annealing (recommended)
- `linear` - Linear decay
- `constant` - Constant LR
- `constant_with_warmup` - Constant after warmup
- `polynomial` - Polynomial decay
- `one_cycle` - OneCycle schedule
- `rex` - REX schedule

### lr_scheduler_kwargs
**Type**: `dict`
**Required**: No
**Default**: None

Additional scheduler arguments.

```yaml
lr_scheduler_kwargs:
  cycle_momentum: false
```

### warmup_ratio
**Type**: `float`
**Required**: No
**Default**: `0.0`

Fraction of training for warmup.

```yaml
warmup_ratio: 0.1  # 10% warmup
```

**Cannot use with** `warmup_steps`.

### warmup_steps
**Type**: `int`
**Required**: No
**Default**: None

Absolute number of warmup steps.

```yaml
warmup_steps: 100
```

**Cannot use with** `warmup_ratio`.

### weight_decay
**Type**: `float`
**Required**: No
**Default**: `0.0`

L2 regularization.

```yaml
weight_decay: 0.01
```

**Typical**: 0.0-0.1

### num_epochs
**Type**: `float`
**Required**: No
**Default**: `1.0`

Number of training epochs.

```yaml
num_epochs: 3
```

### max_steps
**Type**: `int`
**Required**: No
**Default**: None

Maximum training steps (overrides `num_epochs`).

```yaml
max_steps: 10000
```

### max_grad_norm
**Type**: `float`
**Required**: No
**Default**: `1.0`

Gradient clipping max norm.

```yaml
max_grad_norm: 1.0
```

**Typical**: 0.5-5.0

### train_on_inputs
**Type**: `boolean`
**Required**: No
**Default**: `false`

Include user prompts in loss calculation.

```yaml
train_on_inputs: false
```

**Default behavior**: Mask user prompts, only train on assistant responses.

### group_by_length
**Type**: `boolean`
**Required**: No
**Default**: `false`

Group similar-length samples to minimize padding.

```yaml
group_by_length: true
```

**Trade-off**: Slower start, less padding waste.

### lr_quadratic_warmup
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use quadratic warmup for cosine schedule.

```yaml
lr_quadratic_warmup: true
```

### cosine_min_lr_ratio
**Type**: `float`
**Required**: No
**Default**: None

Minimum LR as ratio of max LR.

```yaml
cosine_min_lr_ratio: 0.1  # Decay to 10% of max
```

### cosine_constant_lr_ratio
**Type**: `float`
**Required**: No
**Default**: None

When to start constant LR (as fraction of training).

```yaml
cosine_constant_lr_ratio: 0.8  # Constant after 80%
```

### lr_div_factor
**Type**: `float`
**Required**: No
**Default**: None

Learning rate division factor.

```yaml
lr_div_factor: 25.0
```

### lr_groups
**Type**: `list`
**Required**: No
**Default**: None

Custom learning rates for different modules.

```yaml
lr_groups:
  - name: "embeddings"
    modules: ["embed_tokens", "lm_head"]
    lr: 1e-5
  - name: "attention"
    modules: ["q_proj", "k_proj", "v_proj"]
    lr: 2e-4
```

### embedding_lr
**Type**: `float`
**Required**: No
**Default**: None

Absolute LR for embeddings.

```yaml
embedding_lr: 1e-5
```

### embedding_lr_scale
**Type**: `float`
**Required**: No
**Default**: None

Scale LR for embeddings (relative to main LR).

```yaml
embedding_lr_scale: 0.1  # 10% of main LR
```

### adam_beta1
**Type**: `float`
**Required**: No
**Default**: `0.9`

Adam optimizer beta1.

```yaml
adam_beta1: 0.9
```

### adam_beta2
**Type**: `float`
**Required**: No
**Default**: `0.999`

Adam optimizer beta2.

```yaml
adam_beta2: 0.999
```

### adam_beta3
**Type**: `float`
**Required**: No
**Default**: None

Adam beta3 (CAME optimizer only).

```yaml
adam_beta3: 0.9999
```

### adam_epsilon
**Type**: `float`
**Required**: No
**Default**: `1e-8`

Adam epsilon for numerical stability.

```yaml
adam_epsilon: 1e-8
```

### adam_epsilon2
**Type**: `float`
**Required**: No
**Default**: None

Adam epsilon2 (CAME optimizer only).

```yaml
adam_epsilon2: 1e-12
```

---

## LoRA/PEFT Configuration

Parameters for LoRA and QLoRA training.

### adapter
**Type**: `string`
**Required**: No
**Default**: None

Adapter type to use.

```yaml
adapter: "lora"   # LoRA
adapter: "qlora"  # QLoRA
# Or omit for full fine-tuning
```

### lora_model_dir
**Type**: `string`
**Required**: No
**Default**: None

Path to existing LoRA adapter to load.

```yaml
lora_model_dir: "./adapters/my-lora"
```

### lora_r
**Type**: `int`
**Required**: No (yes for LoRA/QLoRA)
**Default**: None

LoRA rank.

```yaml
lora_r: 32
```

**Typical values**: 8, 16, 32, 64

### lora_alpha
**Type**: `int`
**Required**: No (yes for LoRA/QLoRA)
**Default**: None

LoRA alpha (scaling factor).

```yaml
lora_alpha: 64  # Usually 2× lora_r
```

### lora_dropout
**Type**: `float`
**Required**: No
**Default**: `0.0`

Dropout for LoRA layers.

```yaml
lora_dropout: 0.05
```

**Typical**: 0.0-0.1

### lora_target_modules
**Type**: `list`
**Required**: No
**Default**: Auto-detected

Which modules to apply LoRA to.

```yaml
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### lora_target_linear
**Type**: `boolean`
**Required**: No
**Default**: `false`

Target all linear layers automatically.

```yaml
lora_target_linear: true
```

### lora_target_parameters
**Type**: `list`
**Required**: No
**Default**: None

Regular expression patterns for targeting parameters.

```yaml
lora_target_parameters:
  - ".*proj.*"
```

### lora_modules_to_save
**Type**: `list`
**Required**: No
**Default**: None

Additional modules to save with LoRA.

```yaml
lora_modules_to_save:
  - embed_tokens
  - lm_head
```

**Use when**: You added new tokens to tokenizer.

### lora_fan_in_fan_out
**Type**: `boolean`
**Required**: No
**Default**: `false`

Fan-in/fan-out for LoRA.

```yaml
lora_fan_in_fan_out: true
```

### peft_layers_to_transform
**Type**: `list`
**Required**: No
**Default**: None

Specific layer indices to apply PEFT.

```yaml
peft_layers_to_transform: [0, 1, 2, 3]  # First 4 layers only
```

### peft_use_dora
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use DoRA (Weight-Decomposed LoRA).

```yaml
peft_use_dora: true
```

### peft_use_rslora
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Rank-Stabilized LoRA.

```yaml
peft_use_rslora: true
```

### peft_init_lora_weights
**Type**: `boolean` or `string`
**Required**: No
**Default**: `true`

How to initialize LoRA weights.

```yaml
peft_init_lora_weights: true  # Default (Kaiming uniform)
peft_init_lora_weights: "gaussian"
peft_init_lora_weights: false  # Random
```

### peft_trainable_token_indices
**Type**: `list` or `dict`
**Required**: No
**Default**: None

Train specific embedding tokens.

```yaml
peft_trainable_token_indices: [128000, 128001, 128002]
```

### loraplus_lr_ratio
**Type**: `float`
**Required**: No
**Default**: None

LoRA+ learning rate ratio (lr_B / lr_A).

```yaml
loraplus_lr_ratio: 16.0  # Recommended: 2^4
```

### loraplus_lr_embedding
**Type**: `float`
**Required**: No
**Default**: `1e-6`

LoRA+ LR for embeddings.

```yaml
loraplus_lr_embedding: 1e-6
```

### qlora_sharded_model_loading
**Type**: `boolean`
**Required**: No
**Default**: `false`

Load QLoRA model sharded for FSDP.

```yaml
qlora_sharded_model_loading: true
```

### lora_on_cpu
**Type**: `boolean`
**Required**: No
**Default**: `false`

Load LoRA adapter on CPU (for very large models).

```yaml
lora_on_cpu: true
```

### peft
**Type**: `dict`
**Required**: No
**Default**: None

PEFT configuration (LoftQ, etc.).

```yaml
peft:
  loftq_config:
    loftq_bits: 4
```

---

## Sequence & Packing

Parameters for sequence handling and sample packing.

### sequence_len
**Type**: `int`
**Required**: No
**Default**: `512`

Maximum sequence length.

```yaml
sequence_len: 2048
sequence_len: 4096
sequence_len: 8192
```

**Memory**: Quadratic with length (2x length = 4x memory).

### eval_sequence_len
**Type**: `int`
**Required**: No
**Default**: Same as `sequence_len`

Max sequence length for evaluation.

```yaml
eval_sequence_len: 4096
```

### min_sample_len
**Type**: `int`
**Required**: No
**Default**: None

Minimum sample length (filter shorter samples).

```yaml
min_sample_len: 100
```

### max_prompt_len
**Type**: `int`
**Required**: No
**Default**: None

Maximum prompt length (for RL training).

```yaml
max_prompt_len: 1024
```

### excess_length_strategy
**Type**: `string`
**Required**: No
**Default**: `"drop"`

What to do with samples exceeding `sequence_len`.

```yaml
excess_length_strategy: "drop"      # Remove sample
excess_length_strategy: "truncate"  # Cut to sequence_len
```

### sample_packing
**Type**: `boolean`
**Required**: No
**Default**: `false`

Pack multiple samples into one sequence.

```yaml
sample_packing: true  # RECOMMENDED
```

**Benefits**: Better GPU utilization, faster training.

### sample_packing_group_size
**Type**: `int`
**Required**: No
**Default**: `100000`

Samples to group together for packing.

```yaml
sample_packing_group_size: 100000
```

**Larger = better packing**, but more memory.

### sample_packing_bin_size
**Type**: `int`
**Required**: No
**Default**: `200`

Max samples per packed sequence.

```yaml
sample_packing_bin_size: 200
```

**Increase if**: Long `sequence_len` with many short samples.

### sample_packing_sequentially
**Type**: `boolean`
**Required**: No
**Default**: `false`

Pack samples in original order (for curriculum learning).

```yaml
sample_packing_sequentially: true
```

### sample_packing_mp_start_method
**Type**: `string`
**Required**: No
**Default**: None

Multiprocessing method for packing.

```yaml
sample_packing_mp_start_method: "fork"  # or "spawn", "forkserver"
```

### eval_sample_packing
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use packing for evaluation.

```yaml
eval_sample_packing: true
```

### pad_to_sequence_len
**Type**: `boolean`
**Required**: No
**Default**: Auto (true if `sample_packing`)

Pad to fixed sequence length.

```yaml
pad_to_sequence_len: true
```

**Benefit**: Constant-sized buffers, less memory fragmentation.

### curriculum_sampling
**Type**: `boolean`
**Required**: No
**Default**: `false`

Sequential sampling for curriculum learning.

```yaml
curriculum_sampling: true
```

### multipack_real_batches
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use real batches with multipack.

```yaml
multipack_real_batches: true
```

### batch_flattening
**Type**: `boolean` or `string`
**Required**: No
**Default**: `false`

Use batch flattening (when not using sample_packing).

```yaml
batch_flattening: true
batch_flattening: "auto"
```

---

## Precision & Mixed Precision

Parameters controlling numerical precision.

### bf16
**Type**: `boolean` or `string`
**Required**: No
**Default**: `"auto"`

Use bfloat16 precision.

```yaml
bf16: auto   # Auto-detect (recommended)
bf16: true   # Force enable
bf16: false  # Disable
```

**Requires**: Ampere GPU or newer (A100, H100, etc.)

### fp16
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use float16 precision.

```yaml
fp16: true
```

**Note**: Don't use with `bf16`.

### fp8
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use FP8 mixed precision (TorchAO).

```yaml
fp8: true
```

**Best with**: `torch.compile`

### fp8_enable_fsdp_float8_all_gather
**Type**: `boolean`
**Required**: No
**Default**: `false`

Enable FSDP float8 all-gather for FP8.

```yaml
fp8_enable_fsdp_float8_all_gather: true
```

### bfloat16
**Type**: `boolean`
**Required**: No
**Default**: `false`

No AMP bfloat16.

```yaml
bfloat16: true
```

### float16
**Type**: `boolean`
**Required**: No
**Default**: `false`

No AMP float16.

```yaml
float16: true
```

### float32
**Type**: `boolean`
**Required**: No
**Default**: `false`

Full float32 precision.

```yaml
float32: true
```

### tf32
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use TensorFloat-32 (Ampere+).

```yaml
tf32: true
```

**Benefit**: ~3x speedup on Ampere GPUs with minimal precision loss.

---

## Performance Optimizations

Parameters for training performance.

### gradient_checkpointing
**Type**: `boolean` or `string`
**Required**: No
**Default**: `false`

Trade compute for memory.

```yaml
gradient_checkpointing: true
gradient_checkpointing: "offload"       # CPU offload
gradient_checkpointing: "offload_disk"  # Disk offload
```

**Memory savings**: ~30-40%
**Speed cost**: ~20% slower

### gradient_checkpointing_kwargs
**Type**: `dict`
**Required**: No
**Default**: None

Additional gradient checkpointing args.

```yaml
gradient_checkpointing_kwargs:
  use_reentrant: false
```

**Recommended**: `use_reentrant: false` for better error messages.

### activation_offloading
**Type**: `boolean` or `string`
**Required**: No
**Default**: `false`

Offload activations.

```yaml
activation_offloading: true
activation_offloading: "legacy"
activation_offloading: "disk"
```

### flash_attention
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Flash Attention 2.

```yaml
flash_attention: true  # RECOMMENDED
```

**Requires**: Ampere+ GPU, flash-attn package

**Benefits**: 2-4x faster, less memory.

### xformers_attention
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use xFormers memory-efficient attention.

```yaml
xformers_attention: true
```

### sdp_attention
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use PyTorch scaled dot product attention.

```yaml
sdp_attention: true
```

### s2_attention
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Shifted-Sparse attention (Llama only).

```yaml
s2_attention: true
```

### flex_attention
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use PyTorch flex attention (torch >=2.6.0).

```yaml
flex_attention: true
```

### eager_attention
**Type**: `boolean`
**Required**: No
**Default**: `false`

Force eager attention (disable optimizations).

```yaml
eager_attention: true
```

### attn_implementation
**Type**: `string`
**Required**: No
**Default**: None

Custom attention implementation.

```yaml
attn_implementation: "flash_attention_2"
```

### flash_attn_cross_entropy
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use flash-attn cross entropy.

```yaml
flash_attn_cross_entropy: true
```

### flash_attn_rms_norm
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use flash-attn RMS norm.

```yaml
flash_attn_rms_norm: true
```

### flash_attn_fuse_mlp
**Type**: `boolean`
**Required**: No
**Default**: `false`

Fuse MLP operations.

```yaml
flash_attn_fuse_mlp: true
```

### flash_optimum
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use BetterTransformer optimizations.

```yaml
flash_optimum: true
```

### lora_mlp_kernel
**Type**: `boolean`
**Required**: No
**Default**: `false` (auto-enabled for LoRA)

Custom LoRA MLP kernels.

```yaml
lora_mlp_kernel: true
```

**Auto-enabled** for single-GPU LoRA/QLoRA with `lora_dropout=0`.

### lora_qkv_kernel
**Type**: `boolean`
**Required**: No
**Default**: `false` (auto-enabled for LoRA)

Custom LoRA QKV kernels.

```yaml
lora_qkv_kernel: true
```

### lora_o_kernel
**Type**: `boolean`
**Required**: No
**Default**: `false` (auto-enabled for LoRA)

Custom LoRA output kernels.

```yaml
lora_o_kernel: true
```

### unsloth_cross_entropy_loss
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Unsloth cross entropy.

```yaml
unsloth_cross_entropy_loss: true
```

### unsloth_lora_mlp
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Unsloth LoRA MLP.

```yaml
unsloth_lora_mlp: true
```

**⚠️ Single GPU only**.

### unsloth_lora_qkv
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Unsloth LoRA QKV.

```yaml
unsloth_lora_qkv: true
```

### unsloth_lora_o
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Unsloth LoRA output.

```yaml
unsloth_lora_o: true
```

### unsloth_rms_norm
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Unsloth RMS norm.

```yaml
unsloth_rms_norm: true
```

### unsloth_rope
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use Unsloth RoPE.

```yaml
unsloth_rope: true
```

### chunked_cross_entropy
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use chunked cross entropy for memory efficiency.

```yaml
chunked_cross_entropy: true
```

### chunked_cross_entropy_num_chunks
**Type**: `int`
**Required**: No
**Default**: None

Number of chunks for chunked cross entropy.

```yaml
chunked_cross_entropy_num_chunks: 4
```

### torch_compile
**Type**: `boolean` or `string`
**Required**: No
**Default**: `false`

Use torch.compile.

```yaml
torch_compile: true
torch_compile: "auto"  # Enable if torch >=2.6.0
```

### torch_compile_backend
**Type**: `string`
**Required**: No
**Default**: None

Backend for torch.compile.

```yaml
torch_compile_backend: "inductor"
```

### torch_compile_mode
**Type**: `string`
**Required**: No
**Default**: `"default"`

Compilation mode.

```yaml
torch_compile_mode: "default"
torch_compile_mode: "reduce-overhead"
torch_compile_mode: "max-autotune"
```

### tiled_mlp
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use ALST tiled MLP for long context.

```yaml
tiled_mlp: true
```

### tiled_mlp_num_shards
**Type**: `int`
**Required**: No
**Default**: Auto-calculated

Number of shards for tiled MLP.

```yaml
tiled_mlp_num_shards: 4
```

### neftune_noise_alpha
**Type**: `float`
**Required**: No
**Default**: None

NEFTune noise alpha (add noise to embeddings).

```yaml
neftune_noise_alpha: 5.0
```

**Paper default**: 5.0

---

## FSDP Configuration

Fully Sharded Data Parallel configuration.

### fsdp
**Type**: `list`
**Required**: No (deprecated)
**Default**: None

⚠️ **Deprecated**: Use `fsdp_config` instead.

```yaml
fsdp:
  - full_shard
  - auto_wrap
```

### fsdp_config
**Type**: `dict`
**Required**: No
**Default**: None

FSDP configuration (recommended).

```yaml
fsdp_config:
  fsdp_sharding_strategy: "FULL_SHARD"
  fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
  fsdp_transformer_layer_cls_to_wrap: "LlamaDecoderLayer"
  fsdp_offload_params: false
  fsdp_sync_module_states: true
  fsdp_cpu_ram_efficient_loading: false
  fsdp_use_orig_params: false
  fsdp_limit_all_gathers: false
  fsdp_state_dict_type: "FULL_STATE_DICT"
  activation_checkpointing: false
  reshard_after_forward: false
```

#### fsdp_sharding_strategy
**Type**: `string`
**Options**: `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`

```yaml
fsdp_sharding_strategy: "FULL_SHARD"  # Shard params + gradients + optimizer
fsdp_sharding_strategy: "SHARD_GRAD_OP"  # Shard gradients + optimizer only
fsdp_sharding_strategy: "NO_SHARD"  # DDP equivalent
```

#### fsdp_auto_wrap_policy
**Type**: `string`
**Options**: `TRANSFORMER_BASED_WRAP`, `SIZE_BASED_WRAP`

```yaml
fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"  # Recommended
```

#### fsdp_transformer_layer_cls_to_wrap
**Type**: `string`

Layer class to wrap.

```yaml
# Llama
fsdp_transformer_layer_cls_to_wrap: "LlamaDecoderLayer"

# Mistral
fsdp_transformer_layer_cls_to_wrap: "MistralDecoderLayer"

# Qwen
fsdp_transformer_layer_cls_to_wrap: "Qwen2DecoderLayer"

# DeepSeek
fsdp_transformer_layer_cls_to_wrap: "DeepseekV2DecoderLayer"
```

#### fsdp_offload_params
**Type**: `boolean`

Offload parameters to CPU.

```yaml
fsdp_offload_params: true
```

**Use when**: Model doesn't fit in GPU memory.
**Cost**: Slower training.

#### fsdp_cpu_ram_efficient_loading
**Type**: `boolean`

Load model efficiently to CPU first.

```yaml
fsdp_cpu_ram_efficient_loading: true
```

#### fsdp_sync_module_states
**Type**: `boolean`

Sync module states across ranks.

```yaml
fsdp_sync_module_states: true
```

#### fsdp_use_orig_params
**Type**: `boolean`

Use original parameters (not flattened).

```yaml
fsdp_use_orig_params: false
```

#### fsdp_limit_all_gathers
**Type**: `boolean`

Limit all-gather operations.

```yaml
fsdp_limit_all_gathers: true
```

#### fsdp_state_dict_type
**Type**: `string`
**Options**: `FULL_STATE_DICT`, `LOCAL_STATE_DICT`, `SHARDED_STATE_DICT`

```yaml
fsdp_state_dict_type: "FULL_STATE_DICT"
```

### fsdp_version
**Type**: `int`
**Options**: `1`, `2`

FSDP version (2 requires torch >=2.7.0).

```yaml
fsdp_version: 2
```

---

## DeepSpeed Configuration

### deepspeed
**Type**: `string` or `dict`
**Required**: No
**Default**: None

Path to DeepSpeed config or inline config.

```yaml
# Path to JSON file
deepspeed: "deepspeed_configs/zero3.json"

# Inline
deepspeed:
  gradient_accumulation_steps: 16
  train_micro_batch_size_per_gpu: 1
  zero_optimization:
    stage: 3
```

### deepcompile
**Type**: `boolean`
**Required**: No
**Default**: `false`

Use DeepCompile for faster training.

```yaml
deepcompile: true
```

---

## Distributed & Parallelism

Parameters for distributed training.

### tensor_parallel_size
**Type**: `int`
**Required**: No
**Default**: None

Number of tensor parallel processes.

```yaml
tensor_parallel_size: 2
```

**Use for**: Very large models (>70B) on many GPUs.
**Requires**: DeepSpeed AutoTP.

### context_parallel_size
**Type**: `int`
**Required**: No
**Default**: None

Context/sequence parallelism size.

```yaml
context_parallel_size: 2
```

**Use for**: Very long sequences (>8192) that don't fit in single GPU.

### heads_k_stride
**Type**: `int`
**Required**: No
**Default**: None

Stride across key dimension (context parallelism).

```yaml
heads_k_stride: 2
```

### ring_attn_func
**Type**: `string`
**Required**: No
**Default**: Auto

Ring attention function.

```yaml
ring_attn_func: "varlen_llama3"
ring_attn_func: "batch_ring"
ring_attn_func: "batch_zigzag"
ring_attn_func: "batch_stripe"
```

### dp_shard_size
**Type**: `int`
**Required**: No
**Default**: None

Data parallel shard size.

```yaml
dp_shard_size: 4
```

### dp_replicate_size
**Type**: `int`
**Required**: No
**Default**: None

Data parallel replicate size.

```yaml
dp_replicate_size: 2
```

### ddp
**Type**: `boolean`
**Required**: No
**Default**: Auto-detected

Use DistributedDataParallel.

```yaml
ddp: true
```

**Auto-detected**: true if world_size > 1

### ddp_timeout
**Type**: `int`
**Required**: No
**Default**: None

DDP timeout in seconds.

```yaml
ddp_timeout: 7200  # 2 hours
```

### ddp_bucket_cap_mb
**Type**: `int`
**Required**: No
**Default**: None

DDP bucket size in MB.

```yaml
ddp_bucket_cap_mb: 25
```

### ddp_broadcast_buffers
**Type**: `boolean`
**Required**: No
**Default**: None

Broadcast buffers in DDP.

```yaml
ddp_broadcast_buffers: true
```

### ddp_find_unused_parameters
**Type**: `boolean`
**Required**: No
**Default**: None

Find unused parameters in DDP.

```yaml
ddp_find_unused_parameters: false
```

---

## Evaluation & Checkpointing

Parameters for evaluation and checkpoint saving.

### val_set_size
**Type**: `float`
**Required**: No
**Default**: `0.0`

Validation split (0.0 = no validation, 1.0 = 100%).

```yaml
val_set_size: 0.05  # 5%
```

### eval_steps
**Type**: `int` or `float`
**Required**: No
**Default**: None

Evaluate every N steps (int) or fraction of training (float).

```yaml
eval_steps: 100  # Every 100 steps
eval_steps: 0.1  # Every 10% of training
```

### evals_per_epoch
**Type**: `int`
**Required**: No
**Default**: None

Evaluations per epoch.

```yaml
evals_per_epoch: 2
```

**Mutually exclusive** with `eval_steps`.

### eval_strategy
**Type**: `string`
**Required**: No
**Default**: Auto

Evaluation strategy.

```yaml
eval_strategy: "no"      # No evaluation
eval_strategy: "epoch"   # At end of each epoch
eval_strategy: "steps"   # Every eval_steps
```

### save_steps
**Type**: `int` or `float`
**Required**: No
**Default**: None

Save checkpoint every N steps.

```yaml
save_steps: 500
save_steps: 0.1  # Every 10% of training
```

### saves_per_epoch
**Type**: `int`
**Required**: No
**Default**: None

Checkpoints per epoch.

```yaml
saves_per_epoch: 1
```

### save_strategy
**Type**: `string`
**Required**: No
**Default**: Auto

Checkpoint save strategy.

```yaml
save_strategy: "no"      # No checkpoints
save_strategy: "epoch"   # End of each epoch
save_strategy: "steps"   # Every save_steps
save_strategy: "best"    # When best model
```

### save_total_limit
**Type**: `int`
**Required**: No
**Default**: None

Maximum checkpoints to keep.

```yaml
save_total_limit: 3  # Keep last 3 checkpoints
```

### save_first_step
**Type**: `boolean`
**Required**: No
**Default**: `false`

Save checkpoint after first step.

```yaml
save_first_step: true
```

**Use for**: Testing checkpoint saving works.

### save_only_model
**Type**: `boolean`
**Required**: No
**Default**: `false`

Save only model weights (not optimizer state).

```yaml
save_only_model: true
```

**⚠️ Warning**: Cannot resume training from these checkpoints.

### resume_from_checkpoint
**Type**: `string`
**Required**: No
**Default**: None

Resume from specific checkpoint.

```yaml
resume_from_checkpoint: "./outputs/checkpoint-1000"
```

### auto_resume_from_checkpoints
**Type**: `boolean`
**Required**: No
**Default**: `false`

Auto-resume from last checkpoint in `output_dir`.

```yaml
auto_resume_from_checkpoints: true
```

### load_best_model_at_end
**Type**: `boolean`
**Required**: No
**Default**: `false`

Load best model after training.

```yaml
load_best_model_at_end: true
```

### metric_for_best_model
**Type**: `string`
**Required**: No
**Default**: None

Metric to determine best model.

```yaml
metric_for_best_model: "eval_loss"
```

### greater_is_better
**Type**: `boolean`
**Required**: No
**Default**: None

Whether higher metric is better.

```yaml
metric_for_best_model: "eval_accuracy"
greater_is_better: true
```

### early_stopping_patience
**Type**: `int`
**Required**: No
**Default**: None

Stop after N evals without improvement.

```yaml
early_stopping_patience: 3
```

---

## Monitoring & Logging

### Weights & Biases

```yaml
wandb_project: "my-project"
wandb_entity: "my-team"
wandb_name: "experiment-1"
wandb_run_id: null
wandb_watch: "gradients"  # or "all", "false"
wandb_log_model: "checkpoint"  # or "end", false
wandb_mode: "online"  # or "offline", "disabled"
use_wandb: true
```

### MLflow

```yaml
use_mlflow: true
mlflow_tracking_uri: "https://mlflow.example.com"
mlflow_experiment_name: "llama-training"
mlflow_run_name: "run-001"
hf_mlflow_log_artifacts: false
```

### Comet

```yaml
use_comet: true
comet_api_key: "your-key"
comet_workspace: "my-workspace"
comet_project_name: "my-project"
comet_experiment_key: null
comet_mode: "get_or_create"
comet_online: true
comet_experiment_config: {}
```

### TensorBoard

```yaml
use_tensorboard: true
```

Logs saved to `output_dir/runs/`.

### Logging

```yaml
logging_steps: 10
```

Log metrics every N steps.

### Profiling

```yaml
profiler_steps: 10
profiler_steps_start: 0
```

Profile first 10 steps (for performance analysis).

### Tokens Per Second

```yaml
include_tokens_per_second: true
include_tkps: true
```

Report tokens/second metrics.

---

## Advanced Features

### Special Tokens

```yaml
special_tokens:
  bos_token: "<|begin_of_text|>"
  eos_token: "<|end_of_text|>"
  pad_token: "<|pad|>"
  unk_token: "<|unk|>"
  additional_special_tokens:
    - "<|start_header_id|>"
    - "<|end_header_id|>"
```

### Extra Tokens

```yaml
tokens:
  - "<custom_token_1>"
  - "<custom_token_2>"
```

### Token Overrides

```yaml
added_tokens_overrides:
  128000: "<new_token>"
```

### EOT Tokens

```yaml
eot_tokens:
  - "</s>"
  - "[/INST]"
```

### Default System Message

```yaml
default_system_message: "You are a helpful assistant."
```

### Fix Untrained Tokens

```yaml
fix_untrained_tokens: [128000, 128001]
```

Adjust embeddings for untrained tokens.

### ReLoRA

```yaml
relora: true
relora_steps: 250
relora_prune_ratio: 0.9
relora_cpu_offload: false
```

### Jagged Restarts

```yaml
jagged_restart_steps: 500
jagged_restart_warmup_steps: 50
jagged_restart_anneal_steps: 50
```

### LISA (Layerwise Importance Sampling)

```yaml
lisa_n_layers: 2
lisa_step_interval: 20
lisa_layers_attribute: "model.layers"
```

### Loss Watchdog

```yaml
loss_watchdog_threshold: 5.0
loss_watchdog_patience: 3
```

Stop training if loss exceeds threshold for N steps.

### Garbage Collection

```yaml
gc_steps: 100  # Run GC every 100 steps
gc_steps: -1   # Run on epoch end and before evals
gc_steps: 0    # Disabled
```

### Unfrozen Parameters

```yaml
unfrozen_parameters:
  - ".*lm_head.*"
  - ".*embed_tokens.*"
```

Patterns for parameters to train (freeze rest).

### Custom Trainer

```yaml
trainer_cls: "my_module.MyCustomTrainer"
```

### Dataloader Options

```yaml
dataloader_pin_memory: true
dataloader_num_workers: 4
dataloader_prefetch_factor: 256
dataloader_drop_last: false
```

### Remove Unused Columns

```yaml
remove_unused_columns: true
```

### Use Kernels

```yaml
use_kernels: true
```

Enable custom kernels (e.g., MegaBlocks for MoE).

### Llama 4 MoE

```yaml
llama4_linearized_experts: true
```

---

## RL Training

### RL Type

```yaml
rl: "dpo"  # or "ipo", "kto", "simpo", "orpo", "grpo"
```

### DPO/IPO/SimPO

```yaml
rl: "dpo"
rl_beta: 0.1
dpo_use_weighting: false
dpo_label_smoothing: 0.0
dpo_norm_loss: false
dpo_padding_free: false
dpo_generate_during_eval: false
simpo_gamma: 0.5
```

### ORPO

```yaml
rl: "orpo"
orpo_alpha: 1.0
```

### CPO

```yaml
rl: "cpo"
cpo_alpha: 1.0
rpo_alpha: 1.0
```

### KTO

```yaml
rl: "kto"
kto_desirable_weight: 1.0
kto_undesirable_weight: 1.0
```

### Reward Modeling

```yaml
reward_model: true
process_reward_model: false
center_rewards_coefficient: 0.01
num_labels: 2
```

---

## Multimodal

### Image Processing

```yaml
image_size: 224  # or [224, 224]
image_resize_algorithm: null
```

---

## Output Configuration

### output_dir
**Type**: `string`
**Required**: No
**Default**: `"./model-out"`

Where to save model.

```yaml
output_dir: "/workspace/output"
```

---

## Ray Integration

```yaml
use_ray: true
ray_run_name: "my-run"
ray_num_workers: 4
resources_per_worker:
  GPU: 1
  CPU: 8
```

---

## Plugins

```yaml
plugins:
  - "liger"
  - "grokfast"
  - "densemixer"
```

---

## Complete Example (All Common Parameters)

```yaml
# Model
base_model: "meta-llama/Llama-3.1-8B"
model_type: "LlamaForCausalLM"
tokenizer_type: "AutoTokenizer"
trust_remote_code: false
load_in_4bit: true

# Dataset
datasets:
  - path: "/data/conversations.jsonl"
    type: "chat_template"
chat_template: "llama3"
dataset_prepared_path: "last_run_prepared"
dataset_num_proc: 8

# Training
micro_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 2e-4
lr_scheduler: "cosine"
warmup_ratio: 0.1
weight_decay: 0.01
num_epochs: 3
optimizer: "paged_adamw_8bit"
max_grad_norm: 1.0

# LoRA
adapter: "qlora"
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_linear: true

# Sequence
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

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

# Evaluation
val_set_size: 0.05
evals_per_epoch: 2
saves_per_epoch: 1
save_total_limit: 3

# Logging
logging_steps: 10
wandb_project: "my-project"
wandb_entity: "my-team"

# Output
output_dir: "./outputs/llama-8b-qlora"
hub_model_id: "username/llama-8b-finetuned"
save_safetensors: true
```

---

## Next Steps

- **[Dataset Format](DATASET_FORMAT.md)** - OpenAI format specification
- **[Examples](EXAMPLES.md)** - Complete working configurations
- **[Hyperparameters Guide](HYPERPARAMETERS.md)** - Detailed parameter tuning
