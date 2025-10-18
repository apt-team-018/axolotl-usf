"""
Custom Transformers callbacks for monitoring and checkpointing.
Integrates with MetricsTracker and storage backends.
"""

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from typing import Optional, Dict, Any
import os
from pathlib import Path

from integration.utils.logging_config import get_logger
from integration.monitoring.tracker import MetricsTracker
from integration.storage.base import StorageBackend

logger = get_logger(__name__)


class MetricsLoggingCallback(TrainerCallback):
    """
    Callback to log metrics to all configured monitoring backends.
    Integrates with MetricsTracker for multi-backend logging.
    """

    def __init__(self, metrics_tracker: MetricsTracker, job_id: str):
        """
        Initialize metrics logging callback.

        Args:
            metrics_tracker: MetricsTracker instance
            job_id: Job ID for tracking
        """
        self.metrics_tracker = metrics_tracker
        self.job_id = job_id
        super().__init__()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log metrics at each logging step."""
        if logs and state.global_step > 0:
            # Add step to logs if not present
            if 'step' not in logs:
                logs['step'] = state.global_step

            # Log to all backends
            try:
                self.metrics_tracker.log_metrics(logs, state.global_step)
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log evaluation metrics."""
        if metrics:
            try:
                # Prefix eval metrics
                eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
                self.metrics_tracker.log_metrics(eval_metrics, state.global_step)
            except Exception as e:
                logger.warning(f"Failed to log evaluation metrics: {e}")


class CheckpointUploadCallback(TrainerCallback):
    """
    Callback to automatically upload checkpoints to cloud storage.
    Uploads asynchronously to avoid blocking training.
    """

    def __init__(
        self,
        storage_backend: StorageBackend,
        checkpoint_uri_template: str,
        job_id: str,
        upload_every_n_saves: int = 1
    ):
        """
        Initialize checkpoint upload callback.

        Args:
            storage_backend: Storage backend for uploads
            checkpoint_uri_template: Template for checkpoint URIs (can include {step})
            job_id: Job ID
            upload_every_n_saves: Upload every N checkpoint saves (default: every save)
        """
        self.storage_backend = storage_backend
        self.checkpoint_uri_template = checkpoint_uri_template
        self.job_id = job_id
        self.upload_every_n_saves = upload_every_n_saves
        self.save_count = 0
        super().__init__()

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Upload checkpoint after save."""
        self.save_count += 1

        # Only upload every N saves
        if self.save_count % self.upload_every_n_saves != 0:
            return

        # Get checkpoint directory
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"

        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return

        # Format URI with step
        checkpoint_uri = self.checkpoint_uri_template.format(
            job_id=self.job_id,
            step=state.global_step
        )

        logger.info(f"Uploading checkpoint to {checkpoint_uri}")

        try:
            # Upload checkpoint
            self.storage_backend.upload_checkpoint(checkpoint_dir, checkpoint_uri)
            logger.info(f"✅ Checkpoint uploaded successfully")

            # Log checkpoint event
            from integration.monitoring.tracker import MetricsTracker
            if hasattr(self, 'metrics_tracker'):
                self.metrics_tracker.log_checkpoint({
                    "step": state.global_step,
                    "path": checkpoint_uri,
                    "local_path": checkpoint_dir
                })

        except Exception as e:
            logger.error(f"Failed to upload checkpoint: {e}")


class ProgressCallback(TrainerCallback):
    """
    Callback to log training progress in human-readable format.
    Shows progress bar and ETA.
    """

    def __init__(self, total_steps: Optional[int] = None):
        """
        Initialize progress callback.

        Args:
            total_steps: Total training steps (if known)
        """
        self.total_steps = total_steps
        self.start_time = None
        super().__init__()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Record start time."""
        import time
        self.start_time = time.time()
        logger.info("🚀 Training started")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log progress."""
        if not logs or state.global_step == 0:
            return

        # Calculate progress
        if self.total_steps:
            progress_pct = (state.global_step / self.total_steps) * 100
            remaining_steps = self.total_steps - state.global_step
        else:
            progress_pct = None
            remaining_steps = None

        # Calculate ETA
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            if state.global_step > 0:
                time_per_step = elapsed / state.global_step
                if remaining_steps:
                    eta_seconds = time_per_step * remaining_steps
                    eta_str = f"{eta_seconds/3600:.1f}h"
                else:
                    eta_str = "unknown"
            else:
                eta_str = "calculating..."
        else:
            eta_str = "unknown"

        # Log progress
        loss = logs.get('loss', 'N/A')
        lr = logs.get('learning_rate', 'N/A')

        if progress_pct:
            logger.info(
                f"Step {state.global_step}/{self.total_steps} ({progress_pct:.1f}%) | "
                f"Loss: {loss:.4f} | LR: {lr:.2e} | ETA: {eta_str}"
            )
        else:
            logger.info(
                f"Step {state.global_step} | Loss: {loss:.4f} | LR: {lr:.2e}"
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log completion."""
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            logger.info(f"✅ Training completed in {elapsed/3600:.2f} hours")


class ErrorRecoveryCallback(TrainerCallback):
    """
    Callback to handle errors and attempt recovery.
    """

    def __init__(self, job_id: str):
        """
        Initialize error recovery callback.

        Args:
            job_id: Job ID
        """
        self.job_id = job_id
        self.error_count = 0
        self.max_errors = 3
        super().__init__()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Check for NaN losses and other issues."""
        # Check if loss is NaN
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                import math
                if math.isnan(latest_log['loss']) or math.isinf(latest_log['loss']):
                    self.error_count += 1
                    logger.error(
                        f"⚠️ NaN/Inf loss detected at step {state.global_step} "
                        f"(error {self.error_count}/{self.max_errors})"
                    )

                    if self.error_count >= self.max_errors:
                        logger.critical("Too many NaN losses, stopping training")
                        control.should_training_stop = True
                    else:
                        logger.info("Attempting to continue training...")
