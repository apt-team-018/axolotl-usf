"""
Main lifecycle manager orchestrating all lifecycle components.
CRITICAL: Handles complete job lifecycle from start to cleanup.
"""

import os
import signal
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime

from integration.lifecycle.states import TrainingState, TrainingStateMachine
from integration.lifecycle.persistence import JobStatePersistence
from integration.lifecycle.heartbeat import HeartbeatMonitor
from integration.lifecycle.notifications import EmailNotifier
from integration.lifecycle.webhook import WebhookManager
from integration.lifecycle.checkpoint_manager import CheckpointManager
from integration.lifecycle.retry_controller import RetryController
from integration.storage.manager import StorageManager
from integration.utils.logging_config import get_logger

logger = get_logger(__name__)


class LifecycleManager:
    """
    Orchestrates complete training job lifecycle.
    Handles: state tracking, heartbeats, notifications, retries, webhooks, checkpoints.
    """

    def __init__(self, job_config: Dict[str, Any]):
        """
        Initialize lifecycle manager with full configuration.

        Args:
            job_config: Complete job configuration including lifecycle settings
        """
        self.job_config = job_config
        self.job_id = job_config['job_id']

        # Initialize components
        self.state_machine: Optional[TrainingStateMachine] = None
        self.persistence: Optional[JobStatePersistence] = None
        self.heartbeat_monitor: Optional[HeartbeatMonitor] = None
        self.email_notifier: Optional[EmailNotifier] = None
        self.webhook_manager: Optional[WebhookManager] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.retry_controller: Optional[RetryController] = None

        # Storage backends (separate for dataset, checkpoints, model)
        self.storage_dataset: Optional[Any] = None
        self.storage_checkpoints: Optional[Any] = None
        self.storage_model: Optional[Any] = None

        # Lifecycle config
        self.lifecycle_config = job_config.get('lifecycle', {})

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(f"Lifecycle manager created for job: {self.job_id}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self.on_interrupt()
            sys.exit(130)  # Standard exit code for SIGINT

        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    def initialize(self) -> bool:
        """
        Initialize all lifecycle components.

        Returns:
            True if initialization successful
        """
        logger.info("=" * 80)
        logger.info("Initializing Lifecycle Management")
        logger.info("=" * 80)

        try:
            # 1. Initialize state machine
            self.state_machine = TrainingStateMachine(self.job_id)
            logger.info("✅ State machine initialized")

            # 2. Initialize MongoDB persistence (if configured)
            if self.lifecycle_config.get('state_tracking', {}).get('enabled', True):
                mongo_config = self.lifecycle_config['state_tracking']['mongodb']
                self.persistence = JobStatePersistence(
                    mongo_uri=mongo_config['uri'],
                    database=mongo_config['database'],
                    collection=mongo_config['collection'],
                    job_id=self.job_id
                )

                # Create job record
                self.persistence.create_job(self.job_config)
                logger.info("✅ MongoDB persistence initialized")

            # 3. Initialize storage backends
            self._initialize_storage_backends()

            # 4. Initialize checkpoint manager
            if self.storage_checkpoints:
                checkpoint_config = self.job_config['storage']['checkpoints']
                self.checkpoint_manager = CheckpointManager(
                    job_id=self.job_id,
                    storage_backend=self.storage_checkpoints,
                    base_uri=checkpoint_config['uri'],
                    keep_last_n=checkpoint_config.get('keep_last_n', 3),
                    persistence=self.persistence
                )
                logger.info("✅ Checkpoint manager initialized")

            # 5. Initialize email notifier (if configured)
            if self.lifecycle_config.get('notifications', {}).get('email', {}).get('enabled'):
                email_config = self.lifecycle_config['notifications']['email']
                smtp_config = email_config['smtp']

                self.email_notifier = EmailNotifier(
                    smtp_server=smtp_config['server'],
                    smtp_port=smtp_config['port'],
                    smtp_username=smtp_config.get('username', ''),
                    smtp_password=smtp_config.get('password', ''),
                    from_address=email_config['from'],
                    use_tls=smtp_config.get('use_tls', True)
                )
                logger.info("✅ Email notifier initialized")

            # 6. Initialize webhook manager (if configured)
            if self.lifecycle_config.get('server_deletion', {}).get('enabled'):
                deletion_config = self.lifecycle_config['server_deletion']
                webhook_config = deletion_config.get('webhook', {})

                if webhook_config.get('url'):
                    self.webhook_manager = WebhookManager(
                        webhook_url=webhook_config['url'],
                        method=webhook_config.get('method', 'POST'),
                        headers=webhook_config.get('headers', {}),
                        body_template=webhook_config.get('body', '{}'),
                        retry_attempts=webhook_config.get('retry_attempts', 5)
                    )
                    logger.info("✅ Webhook manager initialized")

            # 7. Initialize retry controller
            retry_config = self.lifecycle_config.get('retry', {})
            if retry_config.get('enabled', True):
                self.retry_controller = RetryController(
                    job_id=self.job_id,
                    max_attempts=retry_config.get('max_attempts', 3),
                    delay_minutes=retry_config.get('delay_minutes', [5, 10, 20]),
                    state_machine=self.state_machine,
                    persistence=self.persistence,
                    checkpoint_manager=self.checkpoint_manager
                )
                logger.info("✅ Retry controller initialized")

            # 8. Initialize heartbeat monitor (if configured)
            if self.lifecycle_config.get('heartbeat', {}).get('enabled', True) and self.persistence:
                heartbeat_config = self.lifecycle_config['heartbeat']
                self.heartbeat_monitor = HeartbeatMonitor(
                    persistence=self.persistence,
                    interval_seconds=heartbeat_config.get('interval_seconds', 60),
                    timeout_seconds=heartbeat_config.get('timeout_seconds', 3600),
                    on_timeout_callback=self.on_timeout
                )
                logger.info("✅ Heartbeat monitor initialized")

            logger.info("=" * 80)
            logger.info("✅ Lifecycle management fully initialized")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.critical(f"Failed to initialize lifecycle management: {e}")
            return False

    def _initialize_storage_backends(self) -> None:
        """Initialize separate storage backends for dataset, checkpoints, and model."""
        storage_config = self.job_config.get('storage', {})

        # Dataset storage
        if 'dataset' in storage_config:
            dataset_config = storage_config['dataset']
            self.storage_dataset = StorageManager.get_backend(
                storage_type=dataset_config['type'],
                **dataset_config.get('credentials', {})
            )
            logger.info(f"Dataset storage: {dataset_config['type']}")

        # Checkpoint storage (CRITICAL - must be persistent)
        if 'checkpoints' in storage_config:
            checkpoint_config = storage_config['checkpoints']
            self.storage_checkpoints = StorageManager.get_backend(
                storage_type=checkpoint_config['type'],
                **checkpoint_config.get('credentials', {})
            )
            logger.info(f"Checkpoint storage: {checkpoint_config['type']} (PERSISTENT)")

        # Model storage
        if 'model' in storage_config:
            model_config = storage_config['model']
            self.storage_model = StorageManager.get_backend(
                storage_type=model_config['type'],
                **model_config.get('credentials', {})
            )
            logger.info(f"Model storage: {model_config['type']}")

    def start_training(self) -> None:
        """Mark training as started and begin heartbeat."""
        if self.state_machine:
            self.state_machine.transition_to(
                TrainingState.RUNNING,
                reason="Training started"
            )

        if self.persistence:
            self.persistence.update_state(TrainingState.RUNNING, "Training started")

        # Start heartbeat monitoring
        if self.heartbeat_monitor:
            self.heartbeat_monitor.start()

        logger.info("🚀 Training marked as RUNNING, heartbeat started")

    def on_success(self, output_dir: str, duration_hours: float) -> None:
        """
        Handle successful training completion.
        CRITICAL: Upload model, send email, trigger deletion.

        Args:
            output_dir: Output directory with final model
            duration_hours: Training duration in hours
        """
        logger.info("=" * 80)
        logger.info("✅ Training SUCCESS - Beginning cleanup workflow")
        logger.info("=" * 80)

        # Update state
        if self.state_machine:
            self.state_machine.transition_to(TrainingState.SUCCESS, "Training completed")
        if self.persistence:
            self.persistence.update_state(TrainingState.SUCCESS, "Training completed")

        # Stop heartbeat
        if self.heartbeat_monitor:
            self.heartbeat_monitor.stop()

        # Upload final model
        model_uri = None
        if self.storage_model:
            try:
                model_config = self.job_config['storage']['model']
                model_uri = model_config['uri']

                logger.info(f"Uploading final model to {model_uri}")
                self.storage_model.upload_model(output_dir, model_uri)
                logger.info("✅ Final model uploaded")

            except Exception as e:
                logger.error(f"Failed to upload final model: {e}")

        # Send success email
        if self.email_notifier:
            try:
                self.email_notifier.send_success_notification(
                    to_addresses=self.lifecycle_config['notifications']['email']['to'],
                    job_id=self.job_id,
                    model=self.job_config['model'],
                    duration_hours=duration_hours,
                    model_uri=model_uri or "Not uploaded",
                    deletion_delay_minutes=self._get_success_deletion_delay()
                )

                if self.persistence:
                    self.persistence.record_notification("success", {"sent": True})

            except Exception as e:
                logger.error(f"Failed to send success email: {e}")

        # Trigger deletion webhook (with delay for upload verification)
        if self.webhook_manager and self._should_delete_on_success():
            delay_minutes = self._get_success_deletion_delay()
            delay_seconds = delay_minutes * 60

            try:
                self.webhook_manager.call_deletion_webhook(
                    job_id=self.job_id,
                    reason="Training completed successfully",
                    status="SUCCESS",
                    delay_seconds=delay_seconds
                )

                if self.persistence:
                    self.persistence.record_webhook_call(
                        webhook_type="deletion",
                        url=self.webhook_manager.webhook_url,
                        status_code=None,  # Will be updated after call
                        response="Scheduled"
                    )

            except Exception as e:
                logger.error(f"Failed to trigger deletion webhook: {e}")

        logger.info("✅ Success workflow complete")

    def on_failure(
        self,
        error: str,
        failed_step: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Handle training failure.
        CRITICAL: Save checkpoint, check retry, send email.

        Args:
            error: Error message
            failed_step: Step where failure occurred
            output_dir: Output directory

        Returns:
            Checkpoint path to resume from if retrying, None otherwise
        """
        logger.error(f"❌ Training FAILED: {error}")

        # Update state
        if self.state_machine:
            self.state_machine.transition_to(
                TrainingState.FAILED,
                reason=error,
                metadata={"failed_step": failed_step}
            )
        if self.persistence:
            self.persistence.update_state(
                TrainingState.FAILED,
                reason=error,
                metadata={"failed_step": failed_step, "error": error}
            )

        # CRITICAL: Save current checkpoint
        checkpoint_uri = None
        if self.checkpoint_manager and output_dir and failed_step:
            try:
                checkpoint_uri = self.checkpoint_manager.force_save_current_state(
                    output_dir=output_dir,
                    current_step=failed_step,
                    reason="failure"
                )
                logger.info(f"✅ Checkpoint saved on failure: {checkpoint_uri}")
            except Exception as e:
                logger.critical(f"Failed to save checkpoint on failure: {e}")

        # Check if we should retry
        if self.retry_controller:
            should_retry, resume_checkpoint_uri, delay_seconds = \
                self.retry_controller.prepare_retry(error, failed_step)

            if should_retry:
                # Send retry notification
                if self.email_notifier:
                    try:
                        self.email_notifier.send_retry_notification(
                            to_addresses=self.lifecycle_config['notifications']['email']['to'],
                            job_id=self.job_id,
                            attempt=self.retry_controller.current_attempt,
                            max_attempts=self.retry_controller.max_attempts,
                            checkpoint_step=failed_step or 0,
                            retry_delay_minutes=delay_seconds // 60
                        )
                    except Exception as e:
                        logger.error(f"Failed to send retry email: {e}")

                # Execute retry
                return self.retry_controller.execute_retry(resume_checkpoint_uri, delay_seconds)

            else:
                # Max retries reached - terminal failure
                return self._handle_terminal_failure(error, failed_step, checkpoint_uri)

        # No retry configured - send failure email and cleanup
        if self.email_notifier:
            try:
                self.email_notifier.send_failure_notification(
                    to_addresses=self.lifecycle_config['notifications']['email']['to'],
                    job_id=self.job_id,
                    model=self.job_config['model'],
                    attempt=1,
                    max_attempts=1,
                    failed_step=failed_step or 0,
                    total_steps=10000,  # TODO: Get from config
                    error_message=error,
                    checkpoint_uri=checkpoint_uri or "No checkpoint",
                    will_retry=False
                )
            except Exception as e:
                logger.error(f"Failed to send failure email: {e}")

        return None

    def _handle_terminal_failure(
        self,
        error: str,
        failed_step: Optional[int],
        checkpoint_uri: Optional[str]
    ) -> None:
        """Handle terminal failure (max retries reached)."""
        logger.critical("🔴 TERMINAL FAILURE - Max retries reached")

        # Update state
        if self.state_machine:
            self.state_machine.transition_to(
                TrainingState.TERMINAL_FAILURE,
                reason="Max retries exceeded"
            )
        if self.persistence:
            self.persistence.update_state(
                TrainingState.TERMINAL_FAILURE,
                reason="Max retries exceeded"
            )

        # Send terminal failure email
        if self.email_notifier:
            try:
                self.email_notifier.send_terminal_failure_notification(
                    to_addresses=self.lifecycle_config['notifications']['email']['to'],
                    job_id=self.job_id,
                    attempts=self.retry_controller.max_attempts if self.retry_controller else 1,
                    final_step=failed_step or 0,
                    total_steps=10000,  # TODO: Get from config
                    checkpoint_uri=checkpoint_uri or "No checkpoint available",
                    error_diagnosis=error,
                    deletion_delay_hours=self._get_failure_deletion_delay_hours()
                )
            except Exception as e:
                logger.error(f"Failed to send terminal failure email: {e}")

        # Trigger delayed deletion webhook
        if self.webhook_manager and self._should_delete_on_failure():
            delay_hours = self._get_failure_deletion_delay_hours()
            delay_seconds = delay_hours * 3600

            try:
                self.webhook_manager.call_deletion_webhook(
                    job_id=self.job_id,
                    reason="Terminal failure after max retries",
                    status="TERMINAL_FAILURE",
                    delay_seconds=delay_seconds
                )
            except Exception as e:
                logger.error(f"Failed to schedule deletion webhook: {e}")

    def on_timeout(self) -> None:
        """Handle heartbeat timeout (process unresponsive for 1+ hour)."""
        logger.critical("⏰ TIMEOUT DETECTED - No heartbeat for 1+ hour")

        # This will be called by heartbeat monitor thread
        # Just update state and prepare for retry
        if self.state_machine:
            self.state_machine.transition_to(
                TrainingState.TIMEOUT,
                reason="No heartbeat for > 1 hour"
            )

    def on_interrupt(self) -> None:
        """Handle Ctrl+C or SIGTERM - graceful shutdown."""
        logger.warning("⚠️ Training interrupted by user")

        # Save current checkpoint if possible
        # This would need to be called from training context
        logger.info("Attempting graceful shutdown...")

    def check_for_resume(self) -> Optional[str]:
        """
        Check if we should resume from a previous run.
        CRITICAL: Works even after server was deleted and recreated.

        Returns:
            Path to checkpoint to resume from, or None
        """
        if not self.persistence or not self.checkpoint_manager:
            return None

        logger.info("Checking for existing job state (resume capability)...")

        # Get job state from MongoDB
        job_state = self.persistence.get_job_state()

        if not job_state:
            logger.info("No previous state found, starting fresh")
            return None

        # Check if job already completed
        if job_state['status'] == TrainingState.SUCCESS.value:
            logger.info("Job already completed successfully, no resume needed")
            return None

        # Check if we have a checkpoint to resume from
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()

        if not latest_checkpoint:
            logger.info("No checkpoint found for resume")
            return None

        checkpoint_uri = latest_checkpoint['uri']
        checkpoint_step = latest_checkpoint['step']

        logger.info(f"📂 Found checkpoint for resume: step {checkpoint_step}, uri: {checkpoint_uri}")

        # Download checkpoint
        try:
            local_checkpoint_dir = f"/workspace/resume/checkpoint-{checkpoint_step}"
            success = self.checkpoint_manager.download_checkpoint_for_resume(
                checkpoint_uri,
                local_checkpoint_dir
            )

            if success:
                logger.info(f"✅ Checkpoint downloaded, will resume from step {checkpoint_step}")
                return local_checkpoint_dir
            else:
                logger.error("Failed to download checkpoint")
                return None

        except Exception as e:
            logger.error(f"Error downloading checkpoint for resume: {e}")
            return None

    def _should_delete_on_success(self) -> bool:
        """Check if server should be deleted on success."""
        return self.lifecycle_config.get('server_deletion', {}) \
            .get('delete_on_success', {}).get('enabled', False)

    def _should_delete_on_failure(self) -> bool:
        """Check if server should be deleted on failure."""
        return self.lifecycle_config.get('server_deletion', {}) \
            .get('delete_on_failure', {}).get('enabled', False)

    def _get_success_deletion_delay(self) -> int:
        """Get deletion delay for success (in minutes)."""
        return self.lifecycle_config.get('server_deletion', {}) \
            .get('delete_on_success', {}).get('delay_minutes', 5)

    def _get_failure_deletion_delay_hours(self) -> int:
        """Get deletion delay for failure (in hours)."""
        return self.lifecycle_config.get('server_deletion', {}) \
            .get('delete_on_failure', {}).get('delay_hours', 1)

    def cleanup(self) -> None:
        """Cleanup lifecycle components."""
        logger.info("Cleaning up lifecycle components...")

        if self.heartbeat_monitor:
            self.heartbeat_monitor.stop()

        if self.persistence:
            self.persistence.close()

        logger.info("✅ Lifecycle cleanup complete")
