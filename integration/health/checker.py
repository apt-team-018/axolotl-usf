"""
Pre-flight health checks to validate environment before training.
Catches configuration and infrastructure issues early.
"""

import os
import socket
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

from integration.utils.logging_config import get_logger

logger = get_logger(__name__)


class HealthCheckResult:
    """Result of a health check."""

    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} - {self.name}: {self.message}"


class PreFlightChecker:
    """Pre-flight validation before training starts."""

    def __init__(self, job_config: Dict, job_id: str):
        """
        Initialize health checker.

        Args:
            job_config: Job configuration dictionary
            job_id: Job identifier
        """
        self.job_config = job_config
        self.job_id = job_id
        self.results: List[HealthCheckResult] = []

    def check_gpu_availability(self) -> HealthCheckResult:
        """Check if GPUs are available and accessible."""
        logger.info("Checking GPU availability...")

        try:
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    "GPU Availability",
                    False,
                    "No CUDA GPUs detected",
                    {"cuda_available": False}
                )

            gpu_count = torch.cuda.device_count()
            expected_gpus = self.job_config.get('gpus_per_node', 1)

            if gpu_count < expected_gpus:
                return HealthCheckResult(
                    "GPU Availability",
                    False,
                    f"Expected {expected_gpus} GPUs but found {gpu_count}",
                    {"expected": expected_gpus, "found": gpu_count}
                )

            # Get GPU info
            gpu_info = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1024**3,
                    "compute_capability": f"{props.major}.{props.minor}"
                })

            return HealthCheckResult(
                "GPU Availability",
                True,
                f"Found {gpu_count} GPU(s)",
                {"gpus": gpu_info}
            )

        except Exception as e:
            return HealthCheckResult(
                "GPU Availability",
                False,
                f"Error checking GPUs: {str(e)}",
                {"error": str(e)}
            )

    def check_storage_access(self) -> HealthCheckResult:
        """Check if storage backend is accessible."""
        logger.info("Checking storage access...")

        try:
            from integration.storage.manager import StorageManager

            storage_config = self.job_config.get('storage', {})
            storage_type = storage_config.get('type')

            if not storage_type:
                return HealthCheckResult(
                    "Storage Access",
                    False,
                    "No storage type specified",
                    {}
                )

            # Create storage backend
            credentials = storage_config.get('credentials', {})

            try:
                backend = StorageManager.get_backend(storage_type, **credentials)

                # Try to list objects (if possible)
                dataset_uri = storage_config.get('dataset_uri')
                if dataset_uri:
                    # Just check if we can create the backend, don't actually list
                    # as dataset might not exist yet
                    pass

                return HealthCheckResult(
                    "Storage Access",
                    True,
                    f"Storage backend '{storage_type}' initialized successfully",
                    {"storage_type": storage_type}
                )

            except Exception as e:
                return HealthCheckResult(
                    "Storage Access",
                    False,
                    f"Failed to initialize storage backend: {str(e)}",
                    {"storage_type": storage_type, "error": str(e)}
                )

        except Exception as e:
            return HealthCheckResult(
                "Storage Access",
                False,
                f"Error checking storage: {str(e)}",
                {"error": str(e)}
            )

    def check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity for multi-node training."""
        logger.info("Checking network connectivity...")

        try:
            nodes = self.job_config.get('nodes', 1)

            if nodes == 1:
                return HealthCheckResult(
                    "Network Connectivity",
                    True,
                    "Single node training, network check skipped",
                    {"nodes": 1}
                )

            # Check if we can resolve master node
            master_addr = os.getenv('MASTER_ADDR', 'localhost')
            master_port = int(os.getenv('MASTER_PORT', '29500'))

            try:
                # Try to resolve hostname
                socket.gethostbyname(master_addr)

                # Try to connect to master port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((master_addr, master_port))
                sock.close()

                if result == 0:
                    return HealthCheckResult(
                        "Network Connectivity",
                        True,
                        f"Successfully connected to {master_addr}:{master_port}",
                        {"master_addr": master_addr, "master_port": master_port}
                    )
                else:
                    # Port not open yet, but hostname resolves
                    return HealthCheckResult(
                        "Network Connectivity",
                        True,
                        f"Hostname {master_addr} resolves (port not yet open)",
                        {"master_addr": master_addr, "master_port": master_port}
                    )

            except socket.gaierror:
                return HealthCheckResult(
                    "Network Connectivity",
                    False,
                    f"Cannot resolve master hostname: {master_addr}",
                    {"master_addr": master_addr}
                )

        except Exception as e:
            return HealthCheckResult(
                "Network Connectivity",
                False,
                f"Error checking network: {str(e)}",
                {"error": str(e)}
            )

    def check_disk_space(self) -> HealthCheckResult:
        """Check if sufficient disk space is available."""
        logger.info("Checking disk space...")

        try:
            output_dir = Path(self.job_config.get('output_dir', '/workspace/output'))
            output_dir.parent.mkdir(parents=True, exist_ok=True)

            # Get disk space
            stat = os.statvfs(output_dir.parent)
            free_gb = (stat.f_bavail * stat.f_frsize) / 1024**3

            # Estimate required space (very rough)
            # Model size + checkpoints + logs ~ 100GB for large models
            required_gb = 100

            if free_gb < required_gb:
                return HealthCheckResult(
                    "Disk Space",
                    False,
                    f"Low disk space: {free_gb:.1f}GB available, {required_gb}GB recommended",
                    {"available_gb": free_gb, "recommended_gb": required_gb}
                )

            return HealthCheckResult(
                "Disk Space",
                True,
                f"Sufficient disk space: {free_gb:.1f}GB available",
                {"available_gb": free_gb}
            )

        except Exception as e:
            return HealthCheckResult(
                "Disk Space",
                False,
                f"Error checking disk space: {str(e)}",
                {"error": str(e)}
            )

    def check_model_accessibility(self) -> HealthCheckResult:
        """Check if model can be accessed."""
        logger.info("Checking model accessibility...")

        try:
            model_name = self.job_config.get('model')

            if not model_name:
                return HealthCheckResult(
                    "Model Accessibility",
                    False,
                    "No model specified",
                    {}
                )

            # Check if it's a local path
            if os.path.exists(model_name):
                return HealthCheckResult(
                    "Model Accessibility",
                    True,
                    f"Local model found at {model_name}",
                    {"model_path": model_name, "type": "local"}
                )

            # Assume it's a HuggingFace model - just validate format
            if '/' in model_name:
                parts = model_name.split('/')
                if len(parts) == 2:
                    return HealthCheckResult(
                        "Model Accessibility",
                        True,
                        f"Model format valid: {model_name}",
                        {"model": model_name, "type": "huggingface"}
                    )

            return HealthCheckResult(
                "Model Accessibility",
                False,
                f"Model format invalid: {model_name}",
                {"model": model_name}
            )

        except Exception as e:
            return HealthCheckResult(
                "Model Accessibility",
                False,
                f"Error checking model: {str(e)}",
                {"error": str(e)}
            )

    def check_python_dependencies(self) -> HealthCheckResult:
        """Check if required Python packages are installed."""
        logger.info("Checking Python dependencies...")

        required_packages = [
            'torch',
            'transformers',
            'peft',
            'accelerate',
            'datasets',
            'pyyaml',
        ]

        missing = []
        versions = {}

        for package in required_packages:
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                versions[package] = version
            except ImportError:
                missing.append(package)

        if missing:
            return HealthCheckResult(
                "Python Dependencies",
                False,
                f"Missing packages: {', '.join(missing)}",
                {"missing": missing, "installed": versions}
            )

        return HealthCheckResult(
            "Python Dependencies",
            True,
            f"All required packages installed",
            {"packages": versions}
        )

    def run_all_checks(self) -> Tuple[bool, List[HealthCheckResult]]:
        """
        Run all health checks.

        Returns:
            Tuple of (all_passed, results)
        """
        logger.info(f"Running pre-flight health checks for job {self.job_id}")

        self.results = []

        # Run all checks
        checks = [
            self.check_python_dependencies,
            self.check_gpu_availability,
            self.check_disk_space,
            self.check_model_accessibility,
            self.check_storage_access,
            self.check_network_connectivity,
        ]

        for check in checks:
            try:
                result = check()
                self.results.append(result)

                if result.passed:
                    logger.info(str(result))
                else:
                    logger.error(str(result))

            except Exception as e:
                logger.exception(f"Health check failed: {check.__name__}")
                self.results.append(HealthCheckResult(
                    check.__name__,
                    False,
                    f"Check threw exception: {str(e)}",
                    {"error": str(e)}
                ))

        # Check if all passed
        all_passed = all(r.passed for r in self.results)

        # Summary
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        logger.info(f"Health check summary: {passed_count}/{total_count} passed")

        return all_passed, self.results
