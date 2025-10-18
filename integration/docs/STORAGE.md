# Storage Configuration Guide

Complete guide to configuring storage backends for datasets, checkpoints, and models.

## Table of Contents
- [Overview](#overview)
- [Quick Comparison](#quick-comparison)
- [S3 (AWS)](#s3-aws)
- [Azure Blob Storage](#azure-blob-storage)
- [Google Cloud Storage (GCS)](#google-cloud-storage-gcs)
- [HuggingFace Hub](#huggingface-hub)
- [Filesystem (Local/Shared)](#filesystem-localshared)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The training system supports multiple storage backends for maximum flexibility:
- **Datasets**: Where training data is downloaded from
- **Checkpoints**: Where periodic checkpoints are saved (critical for resume)
- **Final Model**: Where the trained model is uploaded

### Storage Types

| Backend | Best For | Pros | Cons |
|---------|----------|------|------|
| **S3** | Production, AWS | Scalable, reliable, cheap | Requires AWS account |
| **Azure** | Production, Azure | Enterprise, global | Requires Azure account |
| **GCS** | Production, GCP | Scalable, integrated | Requires GCP account |
| **HuggingFace** | Model sharing | Easy, public/private | Limited to HF ecosystem |
| **Filesystem** | Development, HPC | Simple, fast | Not persistent across servers |

### URI Formats

```yaml
# S3
dataset_uri: "s3://bucket-name/path/to/dataset.jsonl"

# Azure
dataset_uri: "azure://container-name/path/to/dataset.jsonl"

# GCS
dataset_uri: "gs://bucket-name/path/to/dataset.jsonl"

# HuggingFace
dataset_uri: "hf://username/dataset-name"

# Filesystem
dataset_uri: "/data/dataset.jsonl"
dataset_uri: "file:///data/dataset.jsonl"
```

---

## Quick Comparison

### Feature Matrix

| Feature | S3 | Azure | GCS | HuggingFace | Filesystem |
|---------|----|----|-----|-------------|------------|
| Scalability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Reliability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Cost | $ | $$ | $ | Free | $ (storage) |
| Setup Complexity | Medium | Medium | Medium | Easy | Very Easy |
| Resume Support | ✅ | ✅ | ✅ | ✅ | ⚠️ (if shared) |
| Multi-region | ✅ | ✅ | ✅ | ✅ | ❌ |

### Cost Comparison (Approximate)

**For 100GB dataset + 500GB checkpoints**:
- **S3**: ~$15/month (Standard tier)
- **Azure**: ~$20/month (Hot tier)
- **GCS**: ~$14/month (Standard)
- **HuggingFace**: Free (public), ~$9/month (private)
- **Filesystem**: Storage cost only (NVMe/NFS)

---

## S3 (AWS)

Amazon S3 is the recommended storage for production deployments.

### Features
- ✅ Automatic retry with exponential backoff
- ✅ Checksum verification (MD5)
- ✅ Multipart upload for large files (>25MB)
- ✅ Circuit breaker protection
- ✅ IAM role support (no hardcoded credentials)

### Configuration

#### Option 1: IAM Instance Role (RECOMMENDED)

```yaml
storage:
  type: "s3"
  use_instance_role: true  # Use EC2 instance role
  region: "us-east-1"      # Optional
  dataset_uri: "s3://my-bucket/datasets/alpaca.jsonl"
  output_uri: "s3://my-bucket/models/llama3-8b"
  checkpoint_uri: "s3://my-bucket/checkpoints/llama3-8b"
```

**Best Practice**: This is the most secure option for production.

#### Option 2: Secrets Manager

```yaml
storage:
  type: "s3"
  secret_path: "aws/training-credentials"  # Path in secrets manager
  region: "us-east-1"
  dataset_uri: "s3://my-bucket/datasets/alpaca.jsonl"
  output_uri: "s3://my-bucket/models/llama3-8b"
  checkpoint_uri: "s3://my-bucket/checkpoints/llama3-8b"
```

Secret content in Vault/AWS Secrets Manager:
```json
{
  "aws_access_key_id": "AKIA...",
  "aws_secret_access_key": "...",
  "region": "us-east-1"
}
```

#### Option 3: Environment Variables (Development Only)

```yaml
storage:
  type: "s3"
  credentials:
    aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
    aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
    region: "us-east-1"
  dataset_uri: "s3://my-bucket/datasets/alpaca.jsonl"
  output_uri: "s3://my-bucket/models/llama3-8b"
```

**Warning**: Not recommended for production.

### S3-Compatible Services

Works with S3-compatible services (MinIO, DigitalOcean Spaces, etc.):

```yaml
storage:
  type: "s3"
  credentials:
    aws_access_key_id: "YOUR_KEY"
    aws_secret_access_key: "YOUR_SECRET"
    endpoint_url: "https://nyc3.digitaloceanspaces.com"
    region: "us-east-1"
  dataset_uri: "s3://my-space/datasets/data.jsonl"
  output_uri: "s3://my-space/models/output"
```

### IAM Policy (Minimum Permissions)

```json
{
  "Version": "2012-10-17",
  "Statement": [
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
  ]
}
```

### Bucket Organization

**Recommended structure**:
```
my-training-bucket/
├── datasets/
│   ├── finance-qa.jsonl
│   └── customer-support.jsonl
├── checkpoints/
│   ├── llama3-8b-finance/
│   │   ├── checkpoint-100.tar.gz
│   │   ├── checkpoint-200.tar.gz
│   │   └── checkpoint-300.tar.gz
│   └── llama3-70b-customer/
│       └── checkpoint-50.tar.gz
└── models/
    ├── llama3-8b-finance/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── tokenizer.json
    └── llama3-70b-customer/
        └── ...
```

---

## Azure Blob Storage

Microsoft Azure Blob Storage for Azure-based deployments.

### Configuration

```yaml
storage:
  type: "azure"
  credentials:
    account_name: "${AZURE_STORAGE_ACCOUNT}"
    account_key: "${AZURE_STORAGE_KEY}"
    # OR
    connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
  dataset_uri: "azure://container-name/datasets/data.jsonl"
  output_uri: "azure://container-name/models/output"
  checkpoint_uri: "azure://container-name/checkpoints"
```

### Using Managed Identity (Recommended)

```yaml
storage:
  type: "azure"
  use_managed_identity: true  # Use Azure Managed Identity
  credentials:
    account_name: "mystorageaccount"
  dataset_uri: "azure://training-data/datasets/alpaca.jsonl"
  output_uri: "azure://models/llama3-8b"
```

### Storage Tiers

**Choose based on access patterns**:
- **Hot**: Frequent access (checkpoints, active models) - Higher cost
- **Cool**: Infrequent access (archived models) - Lower cost
- **Archive**: Rarely accessed (compliance) - Lowest cost

---

## Google Cloud Storage (GCS)

Google Cloud Storage for GCP deployments.

### Configuration

```yaml
storage:
  type: "gcs"
  credentials:
    project_id: "my-gcp-project"
    credentials_path: "/path/to/service-account.json"
  dataset_uri: "gs://my-bucket/datasets/alpaca.jsonl"
  output_uri: "gs://my-bucket/models/llama3-8b"
  checkpoint_uri: "gs://my-bucket/checkpoints/llama3-8b"
```

### Using Application Default Credentials (Recommended)

```yaml
storage:
  type: "gcs"
  use_default_credentials: true
  credentials:
    project_id: "my-gcp-project"
  dataset_uri: "gs://my-bucket/datasets/alpaca.jsonl"
  output_uri: "gs://my-bucket/models/llama3-8b"
```

### Service Account Permissions

Minimum IAM roles:
- `roles/storage.objectViewer` (read datasets)
- `roles/storage.objectCreator` (write models/checkpoints)

---

## HuggingFace Hub

Use HuggingFace Hub for datasets and model sharing.

### Configuration

```yaml
storage:
  type: "hf"
  credentials:
    hf_token: "${HF_TOKEN}"
  dataset_uri: "hf://username/my-dataset"
  output_uri: "hf://username/my-fine-tuned-model"
```

### Public vs Private

**Public repositories** (free):
```yaml
dataset_uri: "hf://username/public-dataset"
output_uri: "hf://username/public-model"
```

**Private repositories** (requires Pro):
```yaml
dataset_uri: "hf://username/private-dataset"
output_uri: "hf://username/private-model"
```

### Getting HF Token

1. Visit: https://huggingface.co/settings/tokens
2. Create token with `write` access
3. Set environment variable: `export HF_TOKEN=hf_...`

---

## Filesystem (Local/Shared)

Local or shared filesystem storage.

### Local Storage (Development)

```yaml
storage:
  type: "filesystem"
  dataset_uri: "/data/alpaca.jsonl"
  output_uri: "/models/output"
  checkpoint_uri: "/checkpoints"
```

### Shared Filesystem (HPC/Multi-Node)

```yaml
storage:
  type: "filesystem"
  dataset_uri: "/shared/nfs/datasets/alpaca.jsonl"
  output_uri: "/shared/nfs/models/llama3-8b"
  checkpoint_uri: "/shared/nfs/checkpoints/llama3-8b"
```

**Requirements for multi-node**:
- NFS, Lustre, or similar shared filesystem
- Mounted on all nodes
- Same mount path on all nodes
- Sufficient bandwidth (10Gb+ recommended)

### Limitations

⚠️ **Not suitable for**:
- Ephemeral compute instances (AWS Spot, RunPod, etc.)
- Cross-region training
- Automatic resume after server deletion

✅ **Good for**:
- Local development
- HPC clusters with shared storage
- On-premise deployments

---

## Best Practices

### 1. Separate Buckets/Containers

```yaml
# Separate storage for different purposes
storage:
  type: "s3"
  dataset_uri: "s3://training-datasets/..."      # Read-only
  output_uri: "s3://production-models/..."        # Final models
  checkpoint_uri: "s3://training-checkpoints/..." # Temporary
```

**Benefits**:
- Different access policies
- Easier cleanup
- Better cost management

### 2. Lifecycle Policies

**S3 Example** - Delete old checkpoints:
```json
{
  "Rules": [{
    "Id": "DeleteOldCheckpoints",
    "Prefix": "checkpoints/",
    "Status": "Enabled",
    "Expiration": {
      "Days": 30
    }
  }]
}
```

**Azure Example**:
```json
{
  "rules": [{
    "name": "deleteOldCheckpoints",
    "enabled": true,
    "type": "Lifecycle",
    "definition": {
      "filters": {
        "blobTypes": ["blockBlob"],
        "prefixMatch": ["checkpoints/"]
      },
      "actions": {
        "baseBlob": {
          "delete": {
            "daysAfterModificationGreaterThan": 30
          }
        }
      }
    }
  }]
}
```

### 3. Versioning

Enable versioning for models and datasets:
- **S3**: Enable bucket versioning
- **Azure**: Enable blob versioning
- **GCS**: Enable object versioning

### 4. Encryption

**S3** - Server-side encryption:
```yaml
storage:
  type: "s3"
  encryption: "AES256"  # or "aws:kms"
  # ... rest of config
```

**Azure** - Encryption at rest (enabled by default)

**GCS** - Customer-managed encryption keys (CMEK)

### 5. Monitoring

Track storage metrics:
- **S3**: CloudWatch metrics (GetRequests, PutRequests, BytesUploaded)
- **Azure**: Storage Analytics
- **GCS**: Cloud Monitoring

### 6. Cost Optimization

**Tips to reduce costs**:
1. Use cheaper storage tiers for old data
2. Enable lifecycle policies
3. Compress datasets (`.jsonl.gz` instead of `.jsonl`)
4. Delete checkpoints after successful training
5. Use regional storage (not multi-region unless needed)

---

## Troubleshooting

### Connection Errors

**Problem**: Cannot connect to S3/Azure/GCS

**Solutions**:
1. Check credentials are correct
2. Verify network connectivity
3. Check firewall rules
4. Verify bucket/container exists
5. Check IAM permissions

### Checksum Failures

**Problem**: "Checksum verification failed"

**Solutions**:
1. Re-download dataset
2. Check for network issues
3. Verify dataset wasn't corrupted during upload

### Slow Transfers

**Problem**: Very slow upload/download

**Solutions**:
1. Use same region as storage
2. Check network bandwidth
3. Enable multipart upload (automatic for S3)
4. Use faster storage tier (Hot vs Cool)

### Permission Denied

**Problem**: "Access Denied" or "403 Forbidden"

**Solutions**:
1. Verify IAM policy includes required actions
2. Check bucket/container policies
3. Verify credentials haven't expired
4. Check for IP restrictions

### Out of Space

**Problem**: Local disk full during download

**Solutions**:
1. Clean up old checkpoints: `rm -rf /workspace/checkpoints/*`
2. Increase disk size
3. Stream dataset instead of downloading (advanced)

---

## Migration Guide

### From Local to Cloud

```yaml
# Before (local)
storage:
  type: "filesystem"
  dataset_uri: "/data/dataset.jsonl"
  output_uri: "/models/output"

# After (S3)
storage:
  type: "s3"
  use_instance_role: true
  dataset_uri: "s3://my-bucket/datasets/dataset.jsonl"
  output_uri: "s3://my-bucket/models/output"
  checkpoint_uri: "s3://my-bucket/checkpoints"
```

**Steps**:
1. Upload dataset to S3: `aws s3 cp /data/dataset.jsonl s3://my-bucket/datasets/`
2. Update configuration
3. Test with `--validate-only`

### Between Cloud Providers

```bash
# S3 to Azure
aws s3 cp s3://source/dataset.jsonl - | \
  az storage blob upload \
    --account-name myaccount \
    --container-name mycontainer \
    --name dataset.jsonl \
    --file -
```

---

## Next Steps

- **[S3 Detailed Guide](STORAGE_S3.md)** - S3-specific configuration
- **[Secrets Management](SECRETS.md)** - Secure credential handling
- **[Job Configuration](JOB_CONFIG.md)** - Complete config reference
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
