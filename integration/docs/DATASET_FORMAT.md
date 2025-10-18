# Dataset Format Specification

Complete specification for dataset format required by the training system.

## Table of Contents
- [Overview](#overview)
- [Required Format: OpenAI Conversations](#required-format-openai-conversations)
- [Validation Rules](#validation-rules)
- [Format Examples](#format-examples)
- [Creating Your Dataset](#creating-your-dataset)
- [Validation](#validation)
- [Common Errors](#common-errors)

---

## Overview

The training system **ONLY** accepts datasets in **OpenAI conversation format**. This format uses `role` and `content` fields to represent multi-turn conversations.

**Format**: JSONL (JSON Lines) - one conversation per line

**Structure**:
```json
{"conversations": [{"role": "user|assistant|system", "content": "..."}]}
```

---

## Required Format: OpenAI Conversations

### Basic Structure

Each line in your `.jsonl` file must contain:

```json
{
  "conversations": [
    {"role": "system", "content": "System message"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversations` | `list` | ✅ Yes | List of messages in the conversation |
| `conversations[].role` | `string` | ✅ Yes | Must be `system`, `user`, or `assistant` |
| `conversations[].content` | `string` or `list` | ✅ Yes | Message content (cannot be null/empty) |

---

## Validation Rules

### ✅ RULE 1: Use Conversations Key

```json
// ✅ CORRECT
{"conversations": [...]}

// ❌ WRONG
{"messages": [...]}
{"data": [...]}
{"conversation": [...]}
```

### ✅ RULE 2: Valid Roles Only

**Allowed roles**: `system`, `user`, `assistant`

```json
// ✅ CORRECT
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
{"role": "system", "content": "..."}

// ❌ WRONG
{"role": "human", "content": "..."}
{"role": "bot", "content": "..."}
{"role": "ai", "content": "..."}
```

### ✅ RULE 3: Minimum 1 Assistant Message

**Every conversation must have at least one assistant response.**

```json
// ✅ CORRECT - Has assistant
{"conversations": [
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi!"}
]}

// ❌ WRONG - No assistant
{"conversations": [
  {"role": "user", "content": "Hello"}
]}
```

### ✅ RULE 4: Must End with Assistant

**The LAST message must ALWAYS be from assistant.**

```json
// ✅ CORRECT - Ends with assistant
{"conversations": [
  {"role": "user", "content": "Question?"},
  {"role": "assistant", "content": "Answer."}
]}

// ❌ WRONG - Ends with user
{"conversations": [
  {"role": "assistant", "content": "Hi!"},
  {"role": "user", "content": "Bye"}
]}
```

**Why?** The model learns to generate assistant responses. Training data must show complete examples ending with the target output.

### ✅ RULE 5: Content Cannot Be Null or Empty

```json
// ✅ CORRECT
{"role": "user", "content": "Hello"}

// ❌ WRONG
{"role": "user", "content": null}
{"role": "user", "content": ""}
{"role": "user", "content": "   "}
```

### ✅ RULE 6: Content Can Be String or List

**Simple text content**:
```json
{"role": "user", "content": "What is AI?"}
```

**Multimodal content** (text + images):
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
  ]
}
```

**Multimodal types**:
- `text`: Text content
- `image_url`: Image URL or base64 data

---

## Format Examples

### Example 1: Simple QA

```jsonl
{"conversations": [{"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}]}
{"conversations": [{"role": "user", "content": "Explain neural networks"}, {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."}]}
```

### Example 2: With System Message

```jsonl
{"conversations": [{"role": "system", "content": "You are a helpful financial advisor."}, {"role": "user", "content": "Should I invest in stocks?"}, {"role": "assistant", "content": "Investing in stocks depends on your financial goals, risk tolerance, and time horizon. I recommend consulting with a certified financial planner."}]}
```

### Example 3: Multi-Turn Conversation

```jsonl
{"conversations": [{"role": "user", "content": "Hi, I need help with Python"}, {"role": "assistant", "content": "Hello! I'd be happy to help with Python. What do you need assistance with?"}, {"role": "user", "content": "How do I read a file?"}, {"role": "assistant", "content": "You can read a file in Python using:\n\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```\n\nThis automatically closes the file when done."}]}
```

### Example 4: Multimodal (Vision)

```jsonl
{"conversations": [{"role": "user", "content": [{"type": "text", "text": "What's in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}]}, {"role": "assistant", "content": "The image shows a golden retriever playing in a park."}]}
```

### Example 5: Code Generation

```jsonl
{"conversations": [{"role": "system", "content": "You are an expert Python programmer."}, {"role": "user", "content": "Write a function to calculate fibonacci numbers"}, {"role": "assistant", "content": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"}]}
```

---

## Creating Your Dataset

### Method 1: From CSV

```python
import pandas as pd
import json

# Load your data
df = pd.read_csv("qa_pairs.csv")

# Convert to OpenAI format
with open("dataset.jsonl", "w") as f:
    for _, row in df.iterrows():
        conversation = {
            "conversations": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }
        f.write(json.dumps(conversation) + "\n")
```

### Method 2: From Alpaca Format

```python
import json

# Load alpaca format
with open("alpaca.json", "r") as f:
    alpaca_data = json.load(f)

# Convert to OpenAI format
with open("dataset.jsonl", "w") as f:
    for item in alpaca_data:
        # Build user message
        user_msg = item["instruction"]
        if item.get("input"):
            user_msg += f"\n\nInput: {item['input']}"

        conversation = {
            "conversations": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": item["output"]}
            ]
        }
        f.write(json.dumps(conversation) + "\n")
```

### Method 3: From ShareGPT Format

```python
import json

# Load ShareGPT format
with open("sharegpt.json", "r") as f:
    sharegpt_data = json.load(f)

# Convert to OpenAI format
with open("dataset.jsonl", "w") as f:
    for item in sharegpt_data:
        conversations = []

        for msg in item["conversations"]:
            # Map ShareGPT roles to OpenAI
            role_map = {"human": "user", "gpt": "assistant"}
            role = role_map.get(msg["from"], msg["from"])

            conversations.append({
                "role": role,
                "content": msg["value"]
            })

        # Ensure ends with assistant
        if conversations and conversations[-1]["role"] != "assistant":
            continue  # Skip invalid conversations

        f.write(json.dumps({"conversations": conversations}) + "\n")
```

### Method 4: Generate Programmatically

```python
import json

conversations = []

# Example: Generate QA pairs
questions = [
    "What is Python?",
    "How does machine learning work?",
    "Explain quantum computing"
]

answers = [
    "Python is a high-level programming language...",
    "Machine learning uses algorithms to learn from data...",
    "Quantum computing leverages quantum mechanics..."
]

with open("dataset.jsonl", "w") as f:
    for q, a in zip(questions, answers):
        conversation = {
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        }
        f.write(json.dumps(conversation) + "\n")
```

---

## Validation

### Automatic Validation

The system automatically validates your dataset during training:

```bash
python integration/train_wrapper.py --job-config job.yaml
```

**Output**:
```
INFO: Validating dataset format (OpenAI conversation format)...
INFO: ✅ Dataset validation passed: 1000 conversations, 3500 messages
```

### Manual Validation

Validate before training:

```python
from integration.config.dataset_validator import validate_dataset

# Validate with strict mode
result = validate_dataset("dataset.jsonl", strict_mode=True)

if result.is_valid:
    print("✅ Dataset is valid!")
    print(f"Conversations: {result.stats['total_conversations']}")
    print(f"Messages: {result.stats['total_messages']}")
else:
    print("❌ Validation failed:")
    for error in result.errors:
        print(f"  - {error}")
```

### Quick Validation (First 100 Samples)

```python
from integration.config.dataset_validator import quick_validate

# Quick check
is_valid = quick_validate("dataset.jsonl", max_samples=100)
```

### Command-Line Validation

```bash
# Validate dataset
python integration/config/dataset_validator.py dataset.jsonl

# Output:
# ================================================================================
# VALIDATION RESULTS
# ================================================================================
# Valid: True
# Errors: 0
# Warnings: 0
#
# Statistics:
#   total_conversations: 1000
#   total_messages: 3500
#   role_counts: {'system': 100, 'user': 1700, 'assistant': 1700}
```

---

## Common Errors

### Error 1: Missing 'conversations' Key

```json
// ❌ WRONG
{"messages": [...]}

// ✅ CORRECT
{"conversations": [...]}
```

**Fix**: Rename `messages` to `conversations`

### Error 2: Conversation Ends with User

```json
// ❌ WRONG
{"conversations": [
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi!"},
  {"role": "user", "content": "Goodbye"}
]}

// ✅ CORRECT - Add assistant response
{"conversations": [
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi!"},
  {"role": "user", "content": "Goodbye"},
  {"role": "assistant", "content": "Goodbye! Have a great day!"}
]}
```

### Error 3: No Assistant Messages

```json
// ❌ WRONG
{"conversations": [
  {"role": "user", "content": "Question"}
]}

// ✅ CORRECT
{"conversations": [
  {"role": "user", "content": "Question"},
  {"role": "assistant", "content": "Answer"}
]}
```

### Error 4: Empty/Null Content

```json
// ❌ WRONG
{"role": "user", "content": ""}
{"role": "user", "content": null}

// ✅ CORRECT
{"role": "user", "content": "Actual message content"}
```

### Error 5: Invalid Role

```json
// ❌ WRONG
{"role": "human", "content": "..."}
{"role": "bot", "content": "..."}

// ✅ CORRECT
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
```

---

## Configuration

### Job Configuration

```yaml
# job_config.yaml
job_id: "my-training"
model: "meta-llama/Llama-3.1-8B"
training_type: "qlora"

storage:
  type: "s3"
  dataset_uri: "s3://bucket/dataset.jsonl"  # OpenAI format required
  output_uri: "s3://bucket/models/output"

# Dataset configuration
dataset_type: "chat_template"  # FIXED: Always use chat_template

# Chat template (optional - auto-detected from model)
hyperparameters:
  chat_template: "qwen3"  # Model's template (llama3, qwen3, etc.)
  sequence_len: 2048
  sample_packing: true
```

### Chat Template

The `chat_template` parameter specifies which template to use:

**Auto-Detection** (Recommended):
```yaml
# Omit chat_template - system auto-detects from model
hyperparameters:
  # chat_template not specified
  sequence_len: 2048
```

System will detect:
- Llama 3.x → `llama3` template
- Qwen 2.5 → `qwen3` template
- Mistral → `mistral` template
- etc.

**Explicit**:
```yaml
hyperparameters:
  chat_template: "qwen3"  # Force specific template
```

**Available templates**:
- `tokenizer_default` - Use model's built-in template
- `llama3` - Llama 3.x format
- `qwen3` - Qwen 2.5+ format
- `mistral` - Mistral format
- `chatml` - ChatML format
- `alpaca` - Alpaca format
- `phi_3` - Phi-3 format
- `gemma` - Gemma format
- `deepseek_v2` - DeepSeek V2 format
- `jinja` - Custom jinja template (set `chat_template_jinja`)

**Custom Jinja**:
```yaml
hyperparameters:
  chat_template: "jinja"
  chat_template_jinja: |
    {% for message in messages %}
    <|{{ message.role }}|>{{ message.content }}<|end|>
    {% endfor %}
```

---

## Format Examples

### Example 1: Simple QA (Formatted for readability)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    }
  ]
}
```

### Example 2: With System Prompt

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a helpful customer support agent."
    },
    {
      "role": "user",
      "content": "I need to return a product"
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help with your return. Can you provide your order number?"
    }
  ]
}
```

### Example 3: Multi-Turn Dialog

```json
{
  "conversations": [
    {"role": "user", "content": "I'm learning Python"},
    {"role": "assistant", "content": "That's great! What would you like to learn about Python?"},
    {"role": "user", "content": "How do I create a list?"},
    {"role": "assistant", "content": "You can create a list using square brackets: `my_list = [1, 2, 3]` or using the list() constructor."}
  ]
}
```

### Example 4: Long Conversation

```json
{
  "conversations": [
    {"role": "system", "content": "You are a medical advisor. Always recommend consulting a doctor."},
    {"role": "user", "content": "I have a headache"},
    {"role": "assistant", "content": "I'm sorry to hear that. Headaches can have many causes. How long have you had it?"},
    {"role": "user", "content": "About 2 days"},
    {"role": "assistant", "content": "For a headache lasting 2 days, I recommend: 1) Stay hydrated, 2) Rest in a quiet, dark room, 3) If it persists, please consult a doctor."},
    {"role": "user", "content": "Thank you"},
    {"role": "assistant", "content": "You're welcome! Remember to see a doctor if symptoms worsen. Take care!"}
  ]
}
```

### Example 5: Multimodal (Vision Model)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image in detail"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/images/photo.jpg"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": "The image shows a serene mountain landscape with snow-capped peaks reflecting in a crystal-clear alpine lake."
    }
  ]
}
```

### Example 6: Code Generation

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are an expert Python programmer. Provide clean, well-documented code."
    },
    {
      "role": "user",
      "content": "Write a function to calculate the factorial of a number"
    },
    {
      "role": "assistant",
      "content": "```python\ndef factorial(n: int) -> int:\n    \"\"\"\n    Calculate factorial of n.\n    \n    Args:\n        n: Non-negative integer\n    \n    Returns:\n        Factorial of n\n    \"\"\"\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```"
    }
  ]
}
```

---

## Creating Your Dataset

### Step 1: Prepare Your Data

Organize your data with questions and answers.

### Step 2: Convert to OpenAI Format

Use the conversion scripts above or create manually.

### Step 3: Validate

```bash
python integration/config/dataset_validator.py dataset.jsonl
```

### Step 4: Upload to Storage

```bash
# S3
aws s3 cp dataset.jsonl s3://my-bucket/datasets/

# Azure
az storage blob upload \
  --account-name myaccount \
  --container-name datasets \
  --name dataset.jsonl \
  --file dataset.jsonl

# Or use local
cp dataset.jsonl /data/training/
```

### Step 5: Configure Training

```yaml
storage:
  type: "s3"
  dataset_uri: "s3://my-bucket/datasets/dataset.jsonl"
```

---

## Best Practices

### 1. Data Quality

✅ **Do**:
- Use natural, diverse conversations
- Include various conversation lengths
- Cover different topics/scenarios
- Proofread for typos/errors
- Ensure factual accuracy

❌ **Don't**:
- Use repetitive patterns
- Include incomplete conversations
- Mix languages without system prompt
- Include personally identifiable information (PII)

### 2. Conversation Structure

**Good structure**:
- System prompt (optional but recommended)
- Clear user questions
- Comprehensive assistant responses
- Natural dialog flow

### 3. System Prompts

Use system prompts to define behavior:
```json
{"role": "system", "content": "You are a helpful, harmless, and honest assistant."}
{"role": "system", "content": "You are an expert financial advisor. Always recommend consulting professionals."}
{"role": "system", "content": "You are a coding tutor. Explain concepts clearly with examples."}
```

### 4. Dataset Size

**Recommendations**:
- **Minimum**: 100 conversations (for experimentation)
- **Small dataset**: 1,000-10,000 conversations
- **Medium dataset**: 10,000-100,000 conversations
- **Large dataset**: 100,000+ conversations

**For best results**:
- More high-quality data > less high-quality data
- 1,000-10,000 conversations is often sufficient for fine-tuning

### 5. Balance

Balance role distribution:
- User and assistant messages should be roughly equal
- System messages optional (0-10% of conversations)

---

## Troubleshooting

### Validation Failed

**Check**:
1. File exists and is readable
2. File is `.jsonl` format
3. Each line is valid JSON
4. Has `conversations` key (not `messages`)
5. Conversations end with `assistant`
6. No empty/null content

### Slow Validation

For very large datasets (>1M conversations):
```python
# Quick validate first 1000 samples
quick_validate("dataset.jsonl", max_samples=1000)
```

### Memory Issues

For large files:
- Process in chunks
- Use streaming validation
- Split into smaller files

---

## Next Steps

- **[Job Configuration](JOB_CONFIG.md)** - Configure your training job
- **[Examples](EXAMPLES.md)** - Complete training examples
- **[Hyperparameters](HYPERPARAMETERS.md)** - Tune training settings
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
