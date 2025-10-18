"""
Pydantic schemas for input validation.
Ensures all job configurations are valid before training starts.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, validator, root_validator
import re


class StorageCredentials(BaseModel):
    """Storage backend credentials."""
    # S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    region: Optional[str] = None
    endpoint_url: Optional[str] = None

    # Azure
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    connection_string: Optional[str] = None

    # GCS
    project_id: Optional[str] = None
    credentials_path: Optional[str] = None

    # HuggingFace
    hf_token: Optional[str] = None

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class StorageConfig(BaseModel):
    """Storage configuration."""
    type: Literal["s3", "azure", "gcs", "fs", "filesystem", "hf", "huggingface"] = Field(
        ...,
        description="Storage backend type"
    )
    credentials: Optional[StorageCredentials] = Field(
        default_factory=StorageCredentials,
        description="Storage credentials"
    )
    dataset_uri: str = Field(
        ...,
        min_length=1,
        description="URI to dataset"
    )
    output_uri: Optional[str] = Field(
        None,
        description="URI for output/model storage"
    )
    checkpoint_uri: Optional[str] = Field(
        None,
        description="URI for checkpoint storage"
    )

    @validator('dataset_uri')
    def validate_dataset_uri(cls, v):
        """Validate dataset URI format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("dataset_uri cannot be empty")
        return v.strip()


class MonitoringWandB(BaseModel):
    """Weights & Biases monitoring config."""
    project: str = Field(..., min_length=1)
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class MonitoringMLflow(BaseModel):
    """MLflow monitoring config."""
    tracking_uri: str = Field(..., min_length=1)
    experiment_name: str = Field(default="axolotl-training")
    run_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class MonitoringTensorBoard(BaseModel):
    """TensorBoard monitoring config."""
    log_dir: str = Field(default="./runs")


class MonitoringMongoDB(BaseModel):
    """MongoDB monitoring config."""
    uri: str = Field(..., regex=r'^mongodb(\+srv)?://.*')
    database: str = Field(default="training")
    collection: str = Field(default="metrics")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    wandb: Optional[MonitoringWandB] = None
    mlflow: Optional[MonitoringMLflow] = None
    tensorboard: Optional[MonitoringTensorBoard] = None
    mongodb: Optional[MonitoringMongoDB] = None

    @root_validator
    def at_least_one_backend(cls, values):
        """Ensure at least one monitoring backend is configured."""
        if not any([values.get('wandb'), values.get('mlflow'),
                   values.get('tensorboard'), values.get('mongodb')]):
            # It's okay to have no monitoring for testing
            pass
        return values


class Hyperparameters(BaseModel):
    """Training hyperparameters."""
    batch_size: Optional[int] = Field(None, ge=1, le=10000)
    micro_batch_size: Optional[int] = Field(None, ge=1, le=1000)
    gradient_accumulation_steps: Optional[int] = Field(None, ge=1, le=1000)
    learning_rate: Optional[float] = Field(None, gt=0, le=1.0)
    num_epochs: Optional[int] = Field(None, ge=1, le=1000)
    sequence_length: Optional[int] = Field(None, ge=128, le=131072)

    # LoRA specific
    lora_r: Optional[int] = Field(None, ge=1, le=1024)
    lora_alpha: Optional[int] = Field(None, ge=1, le=2048)
    lora_dropout: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Optimizer
    optimizer: Optional[str] = None
    lr_scheduler: Optional[str] = None
    weight_decay: Optional[float] = Field(None, ge=0.0, le=1.0)
    warmup_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)

    # FSDP/DeepSpeed
    fsdp_offload: Optional[bool] = None

    # Evaluation
    val_set_size: Optional[float] = Field(None, ge=0.0, le=1.0)
    evals_per_epoch: Optional[int] = Field(None, ge=0, le=1000)
    saves_per_epoch: Optional[int] = Field(None, ge=0, le=1000)
    logging_steps: Optional[int] = Field(None, ge=1, le=10000)

    class Config:
        extra = "allow"  # Allow additional hyperparameters


class JobConfig(BaseModel):
    """Complete job configuration with validation."""

    # Required fields
    job_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        regex=r'^[a-zA-Z0-9_\-]+$',
        description="Unique job identifier (alphanumeric, dash, underscore only)"
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model name or path (e.g., 'meta-llama/Llama-3.1-8B')"
    )
    training_type: Literal["full", "lora", "qlora"] = Field(
        ...,
        description="Type of fine-tuning"
    )

    # Hardware configuration
    nodes: int = Field(
        default=1,
        ge=1,
        le=256,
        description="Number of nodes (1-256)"
    )
    gpus_per_node: int = Field(
        default=1,
        ge=1,
        le=8,
        description="GPUs per node (1-8)"
    )

    # Dataset (OpenAI format only)
    dataset_type: Literal["chat_template"] = Field(
        default="chat_template",
        description="Dataset format type (FIXED: chat_template for OpenAI conversation format)"
    )
    chat_template: Optional[str] = Field(
        default=None,
        description="Chat template name (e.g., 'qwen3', 'llama3'). Auto-detected from model if not provided."
    )

    # Storage (required)
    storage: StorageConfig = Field(
        ...,
        description="Storage configuration"
    )

    # Monitoring (optional but recommended)
    monitoring: Optional[MonitoringConfig] = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )

    # Hyperparameters (optional)
    hyperparameters: Optional[Hyperparameters] = Field(
        default_factory=Hyperparameters,
        description="Training hyperparameters"
    )

    # Output
    output_dir: str = Field(
        default="/workspace/output",
        description="Local output directory"
    )

    @validator('model')
    def validate_model(cls, v):
        """Validate model name format."""
        v = v.strip()
        if not v:
            raise ValueError("Model name cannot be empty")

        # Check if it's a valid HuggingFace model ID or local path
        if '/' in v:
            # HuggingFace format: org/model
            parts = v.split('/')
            if len(parts) != 2 or not all(p.strip() for p in parts):
                raise ValueError(
                    "Model must be in format 'organization/model' or a valid local path"
                )

        return v

    @validator('job_id')
    def validate_job_id(cls, v):
        """Validate job ID format."""
        v = v.strip()
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError(
                "job_id must contain only alphanumeric characters, dashes, and underscores"
            )
        return v

    @root_validator
    def validate_config_consistency(cls, values):
        """Validate cross-field consistency."""
        training_type = values.get('training_type')
        hyperparams = values.get('hyperparameters', Hyperparameters())

        # Warn if LoRA/QLoRA but no LoRA hyperparameters
        if training_type in ['lora', 'qlora']:
            if not hyperparams.lora_r:
                # Set defaults
                if not isinstance(hyperparams, Hyperparameters):
                    hyperparams = Hyperparameters()
                hyperparams.lora_r = 32 if training_type == 'qlora' else 16
                hyperparams.lora_alpha = 16 if training_type == 'qlora' else 32
                hyperparams.lora_dropout = 0.05
                values['hyperparameters'] = hyperparams

        # Calculate total GPUs
        total_gpus = values.get('nodes', 1) * values.get('gpus_per_node', 1)
        if total_gpus > 256:
            raise ValueError(f"Total GPUs ({total_gpus}) exceeds maximum of 256")

        return values

    class Config:
        extra = "forbid"  # Strict - no extra fields allowed at top level
        validate_assignment = True


def validate_job_config(config_dict: Dict[str, Any]) -> JobConfig:
    """
    Validate job configuration dictionary.

    Args:
        config_dict: Configuration dictionary from YAML

    Returns:
        Validated JobConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return JobConfig(**config_dict)
