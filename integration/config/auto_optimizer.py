"""
Automatic configuration optimizer.
Calculates optimal training configuration based on model, hardware, and user dataset parameters.
"""

import torch
from typing import Dict, Any, Optional, Literal
from transformers import AutoConfig
from pathlib import Path

from integration.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelAnalyzer:
    """Analyzes model to extract metadata and capabilities."""

    @staticmethod
    def analyze_model(model_name_or_path: str) -> Dict[str, Any]:
        """
        Auto-detect model metadata from HuggingFace or local config.

        Args:
            model_name_or_path: Model name or local path

        Returns:
            Model metadata dictionary

        Raises:
            RuntimeError: If model metadata cannot be detected and auto_config is enabled
        """
        logger.info(f"Analyzing model: {model_name_or_path}")

        try:
            # Try to load config
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

            # Calculate total parameters
            num_params = ModelAnalyzer._estimate_parameters(config)

            # Detect MoE
            is_moe = hasattr(config, 'num_local_experts') or hasattr(config, 'num_experts')
            num_experts = getattr(config, 'num_local_experts', getattr(config, 'num_experts', None))
            experts_per_token = getattr(config, 'num_experts_per_tok', getattr(config, 'num_key_value_heads', None))

            metadata = {
                "num_parameters": num_params,
                "num_layers": getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 32)),
                "hidden_size": getattr(config, 'hidden_size', 4096),
                "num_attention_heads": getattr(config, 'num_attention_heads', 32),
                "num_key_value_heads": getattr(config, 'num_key_value_heads', None),
                "intermediate_size": getattr(config, 'intermediate_size', 11008),
                "vocab_size": getattr(config, 'vocab_size', 32000),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', 4096),

                # MoE specific
                "is_moe": is_moe,
                "num_experts": num_experts,
                "experts_per_token": experts_per_token,

                # Architecture info
                "model_type": config.model_type if hasattr(config, 'model_type') else "unknown",
                "architectures": config.architectures if hasattr(config, 'architectures') else []
            }

            logger.info(
                f"✅ Model analyzed: {num_params/1e9:.1f}B params, "
                f"type={metadata['model_type']}, MoE={is_moe}"
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to analyze model: {e}")
            raise RuntimeError(
                f"❌ Cannot auto-detect model metadata for '{model_name_or_path}'\n\n"
                f"Error: {e}\n\n"
                f"Solutions:\n"
                f"1. Use a HuggingFace model (with config.json)\n"
                f"2. Add config.json to your local model directory\n"
                f"3. Provide model_metadata manually in config\n"
                f"4. Disable auto_config (set enabled: false)"
            )

    @staticmethod
    def _estimate_parameters(config) -> int:
        """Estimate total parameters from config."""
        try:
            # Try to get from config if available
            if hasattr(config, 'num_parameters'):
                return config.num_parameters

            # Estimate from architecture
            hidden_size = getattr(config, 'hidden_size', 4096)
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 32))
            vocab_size = getattr(config, 'vocab_size', 32000)
            intermediate_size = getattr(config, 'intermediate_size', 11008)

            # Rough estimation
            # Embedding: vocab_size * hidden_size
            # Each layer: ~4 * hidden_size^2 (attention) + 3 * hidden_size * intermediate_size (FFN)
            # Output head: vocab_size * hidden_size

            embedding_params = vocab_size * hidden_size
            layer_params = (4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size)
            total_layer_params = num_layers * layer_params
            output_params = vocab_size * hidden_size

            total = embedding_params + total_layer_params + output_params

            logger.debug(f"Estimated parameters: {total/1e9:.2f}B")
            return int(total)

        except Exception as e:
            logger.warning(f"Could not estimate parameters: {e}, using 7B as default")
            return 7_000_000_000  # Conservative default


class HardwareAnalyzer:
    """Analyzes hardware capabilities."""

    @staticmethod
    def analyze_hardware(nodes: int, gpus_per_node: int) -> Dict[str, Any]:
        """
        Detect hardware capabilities.

        Args:
            nodes: Number of nodes
            gpus_per_node: GPUs per node

        Returns:
            Hardware capabilities dictionary
        """
        logger.info(f"Analyzing hardware: {nodes} nodes × {gpus_per_node} GPUs")

        total_gpus = nodes * gpus_per_node

        # Detect GPU type and capabilities
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory_bytes = props.total_memory
            gpu_memory_gb = gpu_memory_bytes / (1024**3)
            compute_capability = f"{props.major}.{props.minor}"

            # Detect capabilities
            supports_bf16 = torch.cuda.is_bf16_supported()
            supports_flash_attn = float(compute_capability) >= 8.0  # Ampere or newer
            supports_tf32 = float(compute_capability) >= 8.0

            # Detect GPU type
            if "H100" in gpu_name or "H200" in gpu_name:
                gpu_type = "H100"
            elif "A100" in gpu_name:
                gpu_type = "A100"
            elif "V100" in gpu_name:
                gpu_type = "V100"
            elif "T4" in gpu_name:
                gpu_type = "T4"
            else:
                gpu_type = gpu_name

        else:
            # CPU fallback
            gpu_type = "CPU"
            gpu_memory_gb = 0
            compute_capability = "0.0"
            supports_bf16 = False
            supports_flash_attn = False
            supports_tf32 = False

        hardware_info = {
            "gpu_type": gpu_type,
            "gpu_memory_gb": gpu_memory_gb,
            "compute_capability": compute_capability,
            "total_gpus": total_gpus,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "supports_bf16": supports_bf16,
            "supports_flash_attn": supports_flash_attn,
            "supports_tf32": supports_tf32
        }

        logger.info(
            f"✅ Hardware detected: {gpu_type} ({gpu_memory_gb:.0f}GB), "
            f"{total_gpus} GPUs total, Flash Attn: {supports_flash_attn}"
        )

        return hardware_info


class AutoConfigOptimizer:
    """
    Automatic configuration optimizer.
    Calculates optimal training config based on model + user dataset params + hardware.
    """

    OPTIMIZE_MODES = ["speed", "memory", "quality", "balanced"]

    def __init__(self):
        """Initialize optimizer."""
        self.model_analyzer = ModelAnalyzer()
        self.hardware_analyzer = HardwareAnalyzer()

    def optimize_config(
        self,
        user_config: Dict[str, Any],
        optimize_for: Literal["speed", "memory", "quality", "balanced"] = "balanced"
    ) -> Dict[str, Any]:
        """
        Generate optimal configuration.

        Args:
            user_config: User-provided configuration including:
                - model (required)
                - training_type (required)
                - nodes (required)
                - gpus_per_node (required)
                - sequence_length (REQUIRED - user must provide)
                - sample_packing (REQUIRED - user must provide)
                - pad_to_sequence_len (REQUIRED - user must provide)
            optimize_for: Optimization mode

        Returns:
            Optimized configuration dictionary
        """
        logger.info(f"Starting auto-configuration (optimize_for={optimize_for})")

        # Validate required inputs
        self._validate_required_inputs(user_config)

        # 1. Analyze model
        model_info = self.model_analyzer.analyze_model(user_config['model'])

        # 2. Analyze hardware
        hardware_info = self.hardware_analyzer.analyze_hardware(
            user_config['nodes'],
            user_config['gpus_per_node']
        )

        # 3. Extract user dataset parameters (REQUIRED from user)
        dataset_params = {
            "sequence_length": user_config['sequence_length'],
            "sample_packing": user_config['sample_packing'],
            "pad_to_sequence_len": user_config.get('pad_to_sequence_len', True)
        }

        # 4. Calculate optimal config based on mode
        if optimize_for == "speed":
            optimal = self._optimize_for_speed(model_info, hardware_info, dataset_params, user_config)
        elif optimize_for == "memory":
            optimal = self._optimize_for_memory(model_info, hardware_info, dataset_params, user_config)
        elif optimize_for == "quality":
            optimal = self._optimize_for_quality(model_info, hardware_info, dataset_params, user_config)
        else:  # balanced
            optimal = self._optimize_for_balanced(model_info, hardware_info, dataset_params, user_config)

        # 5. Merge with user config (user values take precedence)
        final_config = self._merge_configs(user_config, optimal)

        logger.info("✅ Auto-configuration complete")
        self._log_config_summary(final_config)

        return final_config

    def _validate_required_inputs(self, user_config: Dict[str, Any]) -> None:
        """Validate that user provided required dataset parameters."""
        required = ['sequence_length', 'sample_packing']
        missing = [k for k in required if k not in user_config]

        if missing:
            raise ValueError(
                f"Missing required parameters for auto-config: {missing}\n"
                f"You must provide:\n"
                f"  - sequence_length: <int>  (max sequence length for your dataset)\n"
                f"  - sample_packing: <bool>  (whether to use sample packing)\n"
            )

    def _optimize_for_speed(
        self,
        model_info: Dict,
        hardware_info: Dict,
        dataset_params: Dict,
        user_config: Dict
    ) -> Dict[str, Any]:
        """Optimize for fastest training time."""
        logger.info("Optimizing for: SPEED (maximize throughput)")

        optimal = {}

        # Maximize batch size
        optimal['micro_batch_size'] = self._calculate_max_batch_size(
            model_info, hardware_info, dataset_params, user_config,
            conservative=False  # Push GPU to limit
        )

        # Minimize gradient accumulation
        target_batch = 64  # Lower for speed
        per_step = optimal['micro_batch_size'] * hardware_info['total_gpus']
        optimal['gradient_accumulation_steps'] = max(1, target_batch // per_step)

        # Optimization flags
        optimal['flash_attention'] = hardware_info['supports_flash_attn']
        optimal['bf16'] = hardware_info['supports_bf16']
        optimal['tf32'] = hardware_info['supports_tf32']
        optimal['gradient_checkpointing'] = False  # Faster without it

        # Optimizer
        optimal['optimizer'] = "adamw_torch_fused"  # Fastest

        # Parallelism (aggressive for speed)
        if model_info['num_parameters'] > 13_000_000_000:
            optimal['fsdp'] = True
            optimal['fsdp_config'] = self._generate_fsdp_config(model_info, hardware_info, offload=False)

            if model_info['num_parameters'] > 70_000_000_000 and hardware_info['total_gpus'] >= 16:
                optimal['tensor_parallel_size'] = 2  # TP for speed

        return optimal

    def _optimize_for_memory(
        self,
        model_info: Dict,
        hardware_info: Dict,
        dataset_params: Dict,
        user_config: Dict
    ) -> Dict[str, Any]:
        """Optimize for minimum memory usage."""
        logger.info("Optimizing for: MEMORY (minimize GPU memory)")

        optimal = {}

        # Minimize batch size
        optimal['micro_batch_size'] = 1

        # Maximize gradient accumulation (compensate for small batch)
        target_batch = 256  # Larger effective batch
        per_step = optimal['micro_batch_size'] * hardware_info['total_gpus']
        optimal['gradient_accumulation_steps'] = max(1, target_batch // per_step)

        # Memory optimizations
        optimal['flash_attention'] = hardware_info['supports_flash_attn']  # Memory efficient
        optimal['bf16'] = hardware_info['supports_bf16']  # Less memory than fp32
        optimal['gradient_checkpointing'] = True  # Trade compute for memory

        # Optimizer (8-bit for memory)
        if user_config['training_type'] == 'qlora':
            optimal['optimizer'] = "paged_adamw_8bit"
        else:
            optimal['optimizer'] = "adamw_8bit"

        # Aggressive memory saving
        if model_info['num_parameters'] > 7_000_000_000:
            optimal['fsdp'] = True
            optimal['fsdp_config'] = self._generate_fsdp_config(model_info, hardware_info, offload=True)

            # CPU offloading for very large models
            if model_info['num_parameters'] > 30_000_000_000:
                optimal['fsdp_config']['fsdp_cpu_ram_efficient_loading'] = True

        return optimal

    def _optimize_for_quality(
        self,
        model_info: Dict,
        hardware_info: Dict,
        dataset_params: Dict,
        user_config: Dict
    ) -> Dict[str, Any]:
        """Optimize for best model quality."""
        logger.info("Optimizing for: QUALITY (best convergence)")

        optimal = {}

        # Moderate batch size (not too large for stability)
        optimal['micro_batch_size'] = self._calculate_max_batch_size(
            model_info, hardware_info, dataset_params, user_config,
            conservative=True,
            quality_mode=True
        )

        # Optimal effective batch size for quality (64 is good for most LLMs)
        target_batch = 64
        per_step = optimal['micro_batch_size'] * hardware_info['total_gpus']
        optimal['gradient_accumulation_steps'] = max(1, target_batch // per_step)

        # Quality optimizations
        optimal['flash_attention'] = hardware_info['supports_flash_attn']
        optimal['bf16'] = hardware_info['supports_bf16']  # Good precision
        optimal['gradient_checkpointing'] = True  # Allow larger batch

        # Best optimizer
        optimal['optimizer'] = "adamw_torch_fused"  # Full precision + fast

        # Conservative parallelism (avoid quality degradation)
        if model_info['num_parameters'] > 13_000_000_000:
            optimal['fsdp'] = True
            optimal['fsdp_config'] = self._generate_fsdp_config(model_info, hardware_info, offload=False)

            # Minimal tensor parallel (only if absolutely needed)
            if model_info['num_parameters'] > 100_000_000_000:
                optimal['tensor_parallel_size'] = 2

        # Quality-specific settings
        optimal['warmup_ratio'] = 0.1  # Longer warmup for stability
        optimal['weight_decay'] = 0.01  # Regularization

        return optimal

    def _optimize_for_balanced(
        self,
        model_info: Dict,
        hardware_info: Dict,
        dataset_params: Dict,
        user_config: Dict
    ) -> Dict[str, Any]:
        """Optimize for balanced speed/memory/quality."""
        logger.info("Optimizing for: BALANCED (best overall)")

        optimal = {}

        # Balanced batch size
        optimal['micro_batch_size'] = self._calculate_max_batch_size(
            model_info, hardware_info, dataset_params, user_config,
            conservative=True
        )

        # Balanced effective batch
        target_batch = 128  # Good balance
        per_step = optimal['micro_batch_size'] * hardware_info['total_gpus']
        optimal['gradient_accumulation_steps'] = max(1, target_batch // per_step)

        # Balanced optimizations
        optimal['flash_attention'] = hardware_info['supports_flash_attn']
        optimal['bf16'] = hardware_info['supports_bf16']
        optimal['gradient_checkpointing'] = model_info['num_parameters'] > 7_000_000_000

        # Balanced optimizer
        optimal['optimizer'] = "adamw_torch_fused"  # Good speed + quality

        # Smart parallelism
        if model_info['num_parameters'] > 13_000_000_000:
            optimal['fsdp'] = True
            optimal['fsdp_config'] = self._generate_fsdp_config(
                model_info, hardware_info,
                offload=(model_info['num_parameters'] > 30_000_000_000)
            )

            if model_info['num_parameters'] > 70_000_000_000 and hardware_info['total_gpus'] >= 16:
                optimal['tensor_parallel_size'] = 2

            if dataset_params['sequence_length'] > 8192 and hardware_info['total_gpus'] >= 16:
                optimal['context_parallel_size'] = 2

        return optimal

    def _calculate_max_batch_size(
        self,
        model_info: Dict,
        hardware_info: Dict,
        dataset_params: Dict,
        user_config: Dict,
        conservative: bool = True,
        quality_mode: bool = False
    ) -> int:
        """
        Calculate maximum safe batch size.

        Args:
            model_info: Model metadata
            hardware_info: Hardware capabilities
            dataset_params: Dataset parameters from user
            user_config: User configuration
            conservative: Use conservative estimate (safer)
            quality_mode: Optimize for quality (smaller batch)

        Returns:
            Optimal micro batch size
        """
        num_params = model_info['num_parameters']
        gpu_memory_gb = hardware_info['gpu_memory_gb']
        sequence_length = dataset_params['sequence_length']
        training_type = user_config['training_type']

        # Estimate model memory (rough calculation)
        if training_type == 'qlora':
            # 4-bit quantization
            model_memory_gb = (num_params * 0.5) / 1e9  # 0.5 bytes per param
        elif training_type == 'lora':
            # 16-bit + adapters
            model_memory_gb = (num_params * 2) / 1e9  # 2 bytes per param
        else:  # full
            # 16-bit
            model_memory_gb = (num_params * 2) / 1e9

        # Estimate activation memory per token
        hidden_size = model_info['hidden_size']
        num_layers = model_info['num_layers']
        activation_memory_per_token = (hidden_size * num_layers * 2) / 1e9  # Rough estimate

        # Available memory for activations
        available_memory = gpu_memory_gb - model_memory_gb - 5  # 5GB buffer

        if available_memory < 0:
            logger.warning(f"Model ({model_memory_gb:.1f}GB) larger than GPU ({gpu_memory_gb:.0f}GB)!")
            return 1

        # Calculate max batch size
        memory_per_sample = activation_memory_per_token * sequence_length
        max_batch = int(available_memory / memory_per_sample)

        # Apply safety factor
        if conservative:
            max_batch = int(max_batch * 0.7)  # 70% of theoretical max
        else:
            max_batch = int(max_batch * 0.9)  # 90% for speed mode

        # Quality mode uses smaller batch
        if quality_mode:
            max_batch = min(max_batch, 2)  # Max 2 for stability

        # Ensure at least 1
        max_batch = max(1, max_batch)

        logger.debug(
            f"Calculated micro_batch_size={max_batch} "
            f"(model={model_memory_gb:.1f}GB, available={available_memory:.1f}GB, "
            f"seq_len={sequence_length})"
        )

        return max_batch

    def _generate_fsdp_config(
        self,
        model_info: Dict,
        hardware_info: Dict,
        offload: bool = False
    ) -> Dict[str, Any]:
        """Generate FSDP configuration."""
        model_type = model_info['model_type']

        # Determine transformer layer class
        layer_class_map = {
            "llama": "LlamaDecoderLayer",
            "mistral": "MistralDecoderLayer",
            "qwen2": "Qwen2DecoderLayer",
            "phi": "PhiDecoderLayer",
            "gemma": "GemmaDecoderLayer"
        }

        layer_class = layer_class_map.get(model_type, "TransformerBlock")

        fsdp_config = {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": layer_class,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_offload_params": offload,
            "fsdp_sync_module_states": True,
            "fsdp_cpu_ram_efficient_loading": offload
        }

        return fsdp_config

    def _merge_configs(self, user_config: Dict, optimal_config: Dict) -> Dict[str, Any]:
        """
        Merge user config with optimal config.
        User values ALWAYS take precedence.

        Args:
            user_config: User-provided values
            optimal_config: Auto-calculated optimal values

        Returns:
            Merged configuration
        """
        final = optimal_config.copy()

        # User values override auto-configured values
        for key, value in user_config.items():
            if value is not None:  # Only override if user actually provided a value
                final[key] = value
                if key in optimal_config and optimal_config[key] != value:
                    logger.debug(f"User override: {key}={value} (auto was {optimal_config[key]})")

        return final

    def _log_config_summary(self, config: Dict) -> None:
        """Log summary of final configuration."""
        logger.info("Configuration Summary:")
        logger.info(f"  micro_batch_size: {config.get('micro_batch_size')}")
        logger.info(f"  gradient_accumulation_steps: {config.get('gradient_accumulation_steps')}")
        logger.info(f"  effective_batch_size: {config.get('micro_batch_size', 1) * config.get('gradient_accumulation_steps', 1) * config.get('world_size', 1)}")
        logger.info(f"  optimizer: {config.get('optimizer')}")
        logger.info(f"  fsdp: {config.get('fsdp', False)}")
        logger.info(f"  tensor_parallel: {config.get('tensor_parallel_size', 1)}")
        logger.info(f"  flash_attention: {config.get('flash_attention', False)}")
        logger.info(f"  bf16: {config.get('bf16', False)}")
