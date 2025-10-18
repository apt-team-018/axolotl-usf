"""
Configuration validator - PRODUCTION READY.
Comprehensive validation with security checks.
"""

import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates Axolotl configurations - Production Ready.

    Features:
    - Required field validation
    - Type checking
    - URI format validation
    - Email address validation
    - Resource limit validation
    - Security checks (no hardcoded credentials)
    """

    @staticmethod
    def validate(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Comprehensive configuration validation.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Required fields
        required_fields = ["base_model", "datasets", "output_dir"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate datasets
        if "datasets" in config:
            errors.extend(ConfigValidator._validate_datasets(config["datasets"]))

        # Validate adapter configuration
        if "adapter" in config:
            errors.extend(ConfigValidator._validate_adapter(config))

        # Validate batch size configuration
        errors.extend(ConfigValidator._validate_batch_config(config))

        # Validate learning rate
        if "learning_rate" in config:
            if config["learning_rate"] <= 0:
                errors.append("'learning_rate' must be positive")
            elif config["learning_rate"] > 1e-2:
                errors.append(
                    f"Learning rate {config['learning_rate']} is very high. "
                    f"Typical range: 1e-5 to 1e-3"
                )

        # Validate epochs
        if "num_epochs" in config:
            if config["num_epochs"] <= 0:
                errors.append("'num_epochs' must be positive")
            elif config["num_epochs"] > 100:
                errors.append(f"'num_epochs' {config['num_epochs']} seems excessive")

        return len(errors) == 0, errors

    @staticmethod
    def validate_job_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate job configuration (integration layer).

        Checks:
        - Storage URIs
        - Email addresses
        - Resource limits
        - Security (no hardcoded credentials)
        """
        errors = []

        # Validate storage configuration
        if 'storage' in config:
            errors.extend(ConfigValidator._validate_storage(config['storage']))

        # Validate monitoring configuration
        if 'monitoring' in config:
            errors.extend(ConfigValidator._validate_monitoring(config['monitoring']))

        # Validate notifications
        if 'lifecycle' in config and 'notifications' in config['lifecycle']:
            errors.extend(
                ConfigValidator._validate_notifications(
                    config['lifecycle']['notifications']
                )
            )

        # Validate resource limits
        errors.extend(ConfigValidator._validate_resources(config))

        # Security validation
        errors.extend(ConfigValidator._validate_security(config))

        return len(errors) == 0, errors

    @staticmethod
    def _validate_datasets(datasets: Any) -> List[str]:
        """Validate datasets configuration."""
        errors = []

        if not isinstance(datasets, list):
            errors.append("'datasets' must be a list")
            return errors

        if len(datasets) == 0:
            errors.append("'datasets' list cannot be empty")
            return errors

        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                errors.append(f"Dataset at index {idx} must be a dictionary")
            elif "path" not in dataset:
                errors.append(f"Dataset at index {idx} missing 'path' field")

        return errors

    @staticmethod
    def _validate_adapter(config: Dict[str, Any]) -> List[str]:
        """Validate adapter configuration."""
        errors = []
        adapter_type = config.get("adapter")

        if adapter_type in ["lora", "qlora"]:
            # Check LoRA-specific fields
            if "lora_r" not in config:
                errors.append("LoRA adapter requires 'lora_r' field")
            if "lora_alpha" not in config:
                errors.append("LoRA adapter requires 'lora_alpha' field")
            if "lora_target_modules" not in config:
                errors.append("LoRA adapter requires 'lora_target_modules' field")

        return errors

    @staticmethod
    def _validate_batch_config(config: Dict[str, Any]) -> List[str]:
        """Validate batch size configuration."""
        errors = []

        if "micro_batch_size" in config and "gradient_accumulation_steps" in config:
            micro_batch = config["micro_batch_size"]
            grad_accum = config["gradient_accumulation_steps"]

            if micro_batch <= 0:
                errors.append("'micro_batch_size' must be positive")
            elif micro_batch > 128:
                errors.append(
                    f"'micro_batch_size' {micro_batch} is very large and may cause OOM. "
                    f"Consider reducing to 1-64"
                )

            if grad_accum <= 0:
                errors.append("'gradient_accumulation_steps' must be positive")

        return errors

    @staticmethod
    def _validate_storage(storage_config: Dict[str, Any]) -> List[str]:
        """Validate storage configuration and URIs."""
        errors = []

        # Validate storage type
        valid_types = ['s3', 'azure', 'gcs', 'filesystem', 'hf', 'huggingface']
        storage_type = storage_config.get('type', '').lower()

        if storage_type and storage_type not in valid_types:
            errors.append(
                f"Invalid storage type: {storage_type}. "
                f"Valid types: {', '.join(valid_types)}"
            )

        # Validate URIs
        uri_fields = ['dataset_uri', 'output_uri']
        if 'checkpoints' in storage_config:
            uri_fields.append('checkpoints.uri')

        for field in uri_fields:
            # Handle nested fields
            if '.' in field:
                parts = field.split('.')
                value = storage_config
                for part in parts:
                    value = value.get(part, {}) if isinstance(value, dict) else None
                uri = value
            else:
                uri = storage_config.get(field)

            if uri:
                try:
                    parsed = urlparse(str(uri))
                    if not parsed.scheme:
                        errors.append(f"Invalid URI format for {field}: {uri} (missing scheme)")
                except Exception as e:
                    errors.append(f"Failed to parse URI for {field}: {e}")

        return errors

    @staticmethod
    def _validate_monitoring(monitoring_config: Dict[str, Any]) -> List[str]:
        """Validate monitoring configuration."""
        errors = []

        # Validate MongoDB config
        if 'mongodb' in monitoring_config:
            mongo_config = monitoring_config['mongodb']
            if 'uri' not in mongo_config:
                errors.append("MongoDB monitoring requires 'uri' field")
            if 'database' not in mongo_config:
                errors.append("MongoDB monitoring requires 'database' field")

        return errors

    @staticmethod
    def _validate_notifications(notifications_config: Dict[str, Any]) -> List[str]:
        """Validate notification configuration."""
        errors = []

        # Validate email configuration
        if 'email' in notifications_config:
            email_config = notifications_config['email']

            if 'to' in email_config:
                if not isinstance(email_config['to'], list):
                    errors.append("Email 'to' field must be a list")
                else:
                    # Validate email addresses
                    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    for email in email_config['to']:
                        if not re.match(email_regex, email):
                            errors.append(f"Invalid email address format: {email}")

            if email_config.get('enabled') and 'smtp' not in email_config:
                errors.append("Email notifications enabled but 'smtp' config missing")

        return errors

    @staticmethod
    def _validate_resources(config: Dict[str, Any]) -> List[str]:
        """Validate resource limits are sane."""
        errors = []

        hyper = config.get('hyperparameters', {})

        # Validate batch size
        batch = hyper.get('micro_batch_size', 1)
        if batch > 128:
            errors.append(
                f"Batch size {batch} is very large and likely to cause OOM. "
                f"Recommended: 1-64"
            )

        # Validate learning rate
        lr = hyper.get('learning_rate', 1e-4)
        if lr > 1e-2:
            errors.append(
                f"Learning rate {lr} is suspiciously high. "
                f"Typical range: 1e-5 to 1e-3"
            )
        elif lr < 1e-7:
            errors.append(
                f"Learning rate {lr} is suspiciously low. "
                f"Training may not converge."
            )

        # Validate sequence length
        seq_len = hyper.get('sequence_length', 2048)
        if seq_len > 32768:
            errors.append(
                f"Sequence length {seq_len} is very long. "
                f"May cause OOM. Consider using sequence parallelism."
            )

        return errors

    @staticmethod
    def _validate_security(config: Dict[str, Any]) -> List[str]:
        """Security validation - check for hardcoded credentials."""
        errors = []

        # Convert config to string for searching
        config_str = str(config).lower()

        # Check for suspicious patterns
        suspicious_patterns = {
            'password': r'["\']password["\']\s*:\s*["\'][^"\']+["\']',
            'secret': r'["\']secret[_a-z]*["\']\s*:\s*["\'][^"\']+["\']',
            'key': r'["\'](?:access_key|secret_key)["\']\s*:\s*["\'][^"\']+["\']',
            'token': r'["\']token["\']\s*:\s*["\'][^"\']+["\']'
        }

        for cred_type, pattern in suspicious_patterns.items():
            if re.search(pattern, config_str):
                # Check if it's using secrets manager
                if 'secret_path' not in config_str and 'use_instance_role' not in config_str:
                    errors.append(
                        f"Possible hardcoded {cred_type} detected. "
                        f"Use secrets_manager instead of hardcoded credentials."
                    )

        return errors
