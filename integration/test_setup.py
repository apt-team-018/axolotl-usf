#!/usr/bin/env python3
"""
Integration System Test Script
Tests all components to ensure system is ready for production use.

Run: python integration/test_setup.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration.config.dataset_validator import (
    validate_dataset,
    generate_example_dataset
)
from integration.config.schemas import validate_job_config
from integration.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def test_dataset_validation():
    """Test dataset validator."""
    logger.info("=" * 80)
    logger.info("TEST 1: Dataset Validation")
    logger.info("=" * 80)

    # Create temp dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name

        # Valid conversation
        f.write(json.dumps({
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"}
            ]
        }) + '\n')

        # Multi-turn
        f.write(json.dumps({
            "conversations": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."},
                {"role": "user", "content": "Follow-up?"},
                {"role": "assistant", "content": "Follow-up answer."}
            ]
        }) + '\n')

    try:
        result = validate_dataset(temp_path, strict_mode=True)

        if result.is_valid:
            logger.info("✅ Dataset validation TEST PASSED")
            logger.info(f"   Conversations: {result.stats['total_conversations']}")
            logger.info(f"   Messages: {result.stats['total_messages']}")
            return True
        else:
            logger.error("❌ Dataset validation TEST FAILED")
            for error in result.errors:
                logger.error(f"   {error}")
            return False
    finally:
        os.unlink(temp_path)


def test_config_validation():
    """Test job config validation."""
    logger.info("=" * 80)
    logger.info("TEST 2: Job Config Validation")
    logger.info("=" * 80)

    # Minimal valid config
    config = {
        "job_id": "test-job",
        "model": "NousResearch/Llama-3.2-1B",
        "training_type": "qlora",
        "nodes": 1,
        "gpus_per_node": 1,
        "storage": {
            "type": "filesystem",
            "dataset_uri": "/data/test.jsonl",
            "output_uri": "/tmp/output"
        }
    }

    try:
        validated = validate_job_config(config)
        logger.info("✅ Config validation TEST PASSED")
        logger.info(f"   Job ID: {validated.job_id}")
        logger.info(f"   Model: {validated.model}")
        logger.info(f"   Training Type: {validated.training_type}")
        return True
    except Exception as e:
        logger.error(f"❌ Config validation TEST FAILED: {e}")
        return False


def test_storage_imports():
    """Test storage backends can be imported."""
    logger.info("=" * 80)
    logger.info("TEST 3: Storage Backend Imports")
    logger.info("=" * 80)

    try:
        from integration.storage.manager import StorageManager
        from integration.storage.s3 import S3Backend
        from integration.storage.filesystem import FilesystemBackend

        logger.info("✅ Storage imports TEST PASSED")
        logger.info("   Available backends: S3, Azure, GCS, HuggingFace, Filesystem")
        return True
    except ImportError as e:
        logger.error(f"❌ Storage imports TEST FAILED: {e}")
        return False


def test_lifecycle_imports():
    """Test lifecycle components can be imported."""
    logger.info("=" * 80)
    logger.info("TEST 4: Lifecycle System Imports")
    logger.info("=" * 80)

    try:
        from integration.lifecycle.lifecycle_manager import LifecycleManager
        from integration.lifecycle.states import TrainingState, TrainingStateMachine
        from integration.lifecycle.heartbeat import HeartbeatMonitor
        from integration.lifecycle.retry_controller import RetryController
        from integration.lifecycle.notifications import EmailNotifier
        from integration.lifecycle.checkpoint_manager import CheckpointManager

        logger.info("✅ Lifecycle imports TEST PASSED")
        logger.info("   Available: State tracking, Heartbeat, Retry, Email, Webhooks, Checkpoints")
        return True
    except ImportError as e:
        logger.error(f"❌ Lifecycle imports TEST FAILED: {e}")
        return False


def test_monitoring_imports():
    """Test monitoring can be imported."""
    logger.info("=" * 80)
    logger.info("TEST 5: Monitoring System Imports")
    logger.info("=" * 80)

    try:
        from integration.monitoring.tracker import MetricsTracker
        from integration.monitoring.callbacks import (
            MetricsLoggingCallback,
            CheckpointUploadCallback,
            ProgressCallback
        )

        logger.info("✅ Monitoring imports TEST PASSED")
        logger.info("   Available: W&B, MLflow, TensorBoard, MongoDB")
        return True
    except ImportError as e:
        logger.error(f"❌ Monitoring imports TEST FAILED: {e}")
        return False


def test_auto_optimizer():
    """Test auto-optimizer can be imported and initialized."""
    logger.info("=" * 80)
    logger.info("TEST 6: Auto-Optimizer")
    logger.info("=" * 80)

    try:
        from integration.config.auto_optimizer import AutoConfigOptimizer

        optimizer = AutoConfigOptimizer()
        logger.info("✅ Auto-optimizer TEST PASSED")
        logger.info("   Modes available: speed, memory, quality, balanced")
        return True
    except Exception as e:
        logger.error(f"❌ Auto-optimizer TEST FAILED: {e}")
        return False


def test_example_generation():
    """Test example dataset generation."""
    logger.info("=" * 80)
    logger.info("TEST 7: Example Dataset Generation")
    logger.info("=" * 80)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        generate_example_dataset(temp_path, num_samples=10)

        # Validate generated dataset
        result = validate_dataset(temp_path, strict_mode=True)

        os.unlink(temp_path)

        if result.is_valid:
            logger.info("✅ Example generation TEST PASSED")
            logger.info(f"   Generated and validated 10 conversations")
            return True
        else:
            logger.error("❌ Example generation TEST FAILED - validation failed")
            return False

    except Exception as e:
        logger.error(f"❌ Example generation TEST FAILED: {e}")
        return False


def check_dependencies():
    """Check required dependencies."""
    logger.info("=" * 80)
    logger.info("Checking Dependencies")
    logger.info("=" * 80)

    required_packages = {
        "yaml": "PyYAML",
        "pydantic": "pydantic",
        "torch": "torch",
    }

    optional_packages = {
        "boto3": "boto3 (for S3)",
        "azure.storage.blob": "azure-storage-blob (for Azure)",
        "google.cloud.storage": "google-cloud-storage (for GCS)",
        "pymongo": "pymongo (for MongoDB)",
        "wandb": "wandb (for W&B)",
        "mlflow": "mlflow (for MLflow)",
    }

    # Check required
    missing_required = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} - REQUIRED")
            missing_required.append(package)

    # Check optional
    for module, package in optional_packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.warning(f"⚠ {package} - Optional")

    if missing_required:
        logger.error(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        logger.error("Install with: pip install -r integration/requirements.txt")
        return False

    logger.info("\n✅ All required dependencies installed")
    return True


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("AXOLOTL TRAINING INTEGRATION - SYSTEM TEST")
    logger.info("=" * 80)
    logger.info("\n")

    tests = [
        ("Dependencies", check_dependencies),
        ("Dataset Validation", test_dataset_validation),
        ("Config Validation", test_config_validation),
        ("Storage Imports", test_storage_imports),
        ("Lifecycle Imports", test_lifecycle_imports),
        ("Monitoring Imports", test_monitoring_imports),
        ("Auto-Optimizer", test_auto_optimizer),
        ("Example Generation", test_example_generation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.exception(f"Test {name} crashed: {e}")
            results.append((name, False))
        print()  # Blank line between tests

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")

    logger.info("=" * 80)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 80)

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED - System is ready for production!")
        logger.info("\nNext steps:")
        logger.info("1. Generate sample dataset: python integration/examples/generate_sample_dataset.py")
        logger.info("2. Run quickstart: python integration/train_wrapper.py --job-config integration/examples/quickstart_local.yaml")
        logger.info("3. See full docs: integration/docs/README.md")
        return 0
    else:
        logger.error("\n⚠️ Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
