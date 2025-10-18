"""Configuration manager for dynamic config generation."""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml


class ConfigurationManager:
    """
    Dynamically generates Axolotl configurations based on job specifications.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            templates_dir: Directory containing config templates
        """
        if templates_dir is None:
            # Default to templates directory next to this file
            self.templates_dir = Path(__file__).parent / "templates"
        else:
            self.templates_dir = Path(templates_dir)

    def generate_config(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Axolotl config from job specification.

        Args:
            job_spec: Job specification dictionary containing:
                - model: Model name/path
                - training_type: "full", "lora", or "qlora"
                - nodes: Number of nodes
                - gpus_per_node: GPUs per node
                - dataset_path: Local path to dataset
                - batch_size: Total batch size
                - learning_rate: Learning rate
                - num_epochs: Number of epochs
                - output_dir: Output directory
                - ... (other optional parameters)

        Returns:
            Complete Axolotl configuration dictionary
        """
        # Load base config
        base_config = self._load_base_config(job_spec["training_type"])

        # Get hardware config
        hardware_config = self._get_hardware_config(
            job_spec.get("nodes", 1),
            job_spec.get("gpus_per_node", 1)
        )

        # Prepare variable substitutions
        variables = self._prepare_variables(job_spec)

        # Merge configs
        merged_config = self._merge_configs([base_config, hardware_config])

        # Substitute variables
        final_config = self._substitute_variables(merged_config, variables)

        # Post-process config
        final_config = self._post_process_config(final_config, job_spec)

        return final_config

    def _load_base_config(self, training_type: str) -> Dict[str, Any]:
        """Load base configuration template."""
        type_map = {
            "full": "sft_full.yaml",
            "lora": "sft_lora.yaml",
            "qlora": "sft_qlora.yaml"
        }

        template_file = type_map.get(training_type.lower())
        if not template_file:
            raise ValueError(
                f"Unknown training type: {training_type}. "
                f"Must be one of: {', '.join(type_map.keys())}"
            )

        template_path = self.templates_dir / "base" / template_file
        return self._load_yaml_template(template_path)

    def _get_hardware_config(self, nodes: int, gpus_per_node: int) -> Dict[str, Any]:
        """Get hardware-specific configuration."""
        total_gpus = nodes * gpus_per_node

        config = {}

        # Multi-node configuration
        if nodes > 1:
            template_path = self.templates_dir / "hardware" / "multi_node_fsdp.yaml"
            if template_path.exists():
                config = self._load_yaml_template(template_path)
                config["world_size"] = total_gpus

        return config

    def _prepare_variables(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variable substitutions."""
        model_name = job_spec.get("model", "")

        # Determine LoRA target modules based on model architecture
        lora_targets = self._get_lora_targets(model_name)

        # Determine transformer layer class for FSDP
        transformer_class = self._get_transformer_class(model_name)

        variables = {
            "MODEL_NAME": job_spec.get("model", ""),
            "DATASET_PATH": job_spec.get("dataset_path", ""),
            "DATASET_TYPE": job_spec.get("dataset_type", "alpaca"),
            "SEQUENCE_LENGTH": job_spec.get("sequence_length", 4096),
            "MICRO_BATCH_SIZE": job_spec.get("micro_batch_size", 2),
            "GRAD_ACCUM_STEPS": job_spec.get("gradient_accumulation_steps", 4),
            "LEARNING_RATE": job_spec.get("learning_rate", 2e-5),
            "NUM_EPOCHS": job_spec.get("num_epochs", 3),
            "OUTPUT_DIR": job_spec.get("output_dir", "./outputs"),
            "LORA_TARGET_MODULES": lora_targets,
            "TRANSFORMER_LAYER_CLASS": transformer_class,
            "OPTIMIZER": job_spec.get("optimizer", "paged_adamw_8bit"),
            "LR_SCHEDULER": job_spec.get("lr_scheduler", "cosine"),
            "VAL_SET_SIZE": job_spec.get("val_set_size", 0.05),
            "EVALS_PER_EPOCH": job_spec.get("evals_per_epoch", 2),
            "SAVES_PER_EPOCH": job_spec.get("saves_per_epoch", 1),
            "LOGGING_STEPS": job_spec.get("logging_steps", 10),
            "WEIGHT_DECAY": job_spec.get("weight_decay", 0.0),
            "WARMUP_RATIO": job_spec.get("warmup_ratio", 0.1),
            "PAD_TOKEN": job_spec.get("pad_token", "<|end_of_text|>"),
            "LORA_R": job_spec.get("lora_r", 32 if job_spec.get("training_type") == "qlora" else 16),
            "LORA_ALPHA": job_spec.get("lora_alpha", 16 if job_spec.get("training_type") == "qlora" else 32),
            "LORA_DROPOUT": job_spec.get("lora_dropout", 0.05),
            "FSDP_OFFLOAD": job_spec.get("fsdp_offload", False),
            "WORLD_SIZE": job_spec.get("nodes", 1) * job_spec.get("gpus_per_node", 1),
        }

        return variables

    def _get_lora_targets(self, model_name: str) -> List[str]:
        """Get LoRA target modules based on model architecture."""
        model_lower = model_name.lower()

        # Common targets for most transformer models
        common_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        if "llama" in model_lower or "mistral" in model_lower or "qwen" in model_lower:
            return common_targets
        elif "phi" in model_lower:
            return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif "gemma" in model_lower:
            return common_targets
        else:
            # Default to common targets
            return common_targets

    def _get_transformer_class(self, model_name: str) -> str:
        """Get transformer layer class name for FSDP wrapping."""
        model_lower = model_name.lower()

        if "llama" in model_lower:
            return "LlamaDecoderLayer"
        elif "mistral" in model_lower:
            return "MistralDecoderLayer"
        elif "qwen" in model_lower:
            return "Qwen2DecoderLayer"
        elif "phi" in model_lower:
            return "PhiDecoderLayer"
        elif "gemma" in model_lower:
            return "GemmaDecoderLayer"
        else:
            return "TransformerBlock"  # Generic fallback

    def _load_yaml_template(self, path: Path) -> Dict[str, Any]:
        """Load YAML template file."""
        if not path.exists():
            return {}

        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged = {}

        for config in configs:
            if not config:
                continue
            merged = self._deep_merge(merged, config)

        return merged

    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _substitute_variables(self, config: Any, variables: Dict[str, Any]) -> Any:
        """Recursively substitute variables in config."""
        if isinstance(config, dict):
            return {k: self._substitute_variables(v, variables) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_variables(item, variables) for item in config]
        elif isinstance(config, str):
            return self._substitute_string(config, variables)
        else:
            return config

    def _substitute_string(self, text: str, variables: Dict[str, Any]) -> Any:
        """Substitute variables in a string."""
        # Handle ${VAR:default} syntax
        pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'

        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2)

            if var_name in variables:
                value = variables[var_name]
                # Return appropriate type
                if isinstance(value, (int, float, bool, list)):
                    return value
                return str(value)
            elif default_value is not None:
                # Try to convert default to appropriate type
                return self._convert_type(default_value)
            else:
                return match.group(0)  # Keep original if no default

        # Check if entire string is a variable
        full_match = re.fullmatch(pattern, text)
        if full_match:
            result = replace(full_match)
            return result if not isinstance(result, str) else text

        # Otherwise do string substitution
        return re.sub(pattern, lambda m: str(replace(m)), text)

    def _convert_type(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        value_lower = value.lower()

        if value_lower == "true":
            return True
        elif value_lower == "false":
            return False
        elif value_lower == "null" or value_lower == "none":
            return None

        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def _post_process_config(self, config: Dict[str, Any], job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process configuration after substitution."""
        # Add any additional job-specific configs
        if "wandb_project" in job_spec:
            config["wandb_project"] = job_spec["wandb_project"]
        if "wandb_entity" in job_spec:
            config["wandb_entity"] = job_spec["wandb_entity"]

        # Ensure dataset is a list
        if "datasets" in config and not isinstance(config["datasets"], list):
            config["datasets"] = [config["datasets"]]

        return config
