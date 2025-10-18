"""
Auto-retry controller with checkpoint resume.
Orchestrates retry logic with exponential backoff and checkpoint recovery.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from integration.lifecycle.states import TrainingState, TrainingStateMachine
from integration.lifecycle.persistence import JobStatePersistence
from integration.lifecycle.checkpoint_manager import CheckpointManager
from integration.utils.logging_config import get_logger

logger = get_logger(__name__)


class RetryController:
    """
    Controls automatic retry logic with intelligent checkpoint resume.
    Prevents retry loops by detecting identical errors.
    """

    def __init__(
        self,
        job_id: str,
        max_attempts: int,
        delay_minutes: List[int],
        state_machine: TrainingStateMachine,
        persistence: JobStatePersistence,
        checkpoint_manager: CheckpointManager
    ):
        """
        Initialize retry controller.

        Args:
            job_id: Job identifier
            max_attempts: Maximum retry attempts (e.g., 3)
            delay_minutes: Delay between retries [5, 10, 20] minutes
            state_machine: Training state machine
            persistence: MongoDB persistence
            checkpoint_manager: Checkpoint manager
        """
        self.job_id = job_id
        self.max_attempts = max_attempts
        self.delay_minutes = delay_minutes
        self.state_machine = state_machine
        self.persistence = persistence
        self.checkpoint_manager = checkpoint_manager

        self.current_attempt = 1
        self.error_history: List[str] = []

        logger.info(
            f"Retry controller initialized: max_attempts={max_attempts}, "
            f"delays={delay_minutes}"
        )

    def should_retry(self, error: str) -> tuple[bool, Optional[str]]:
        """
        Determine if we should retry based on current state.

        Args:
            error: Error message from failure

        Returns:
            Tuple of (should_retry, reason_if_not)
        """
        # Check if we've exceeded max attempts
        if self.current_attempt >= self.max_attempts:
            reason = f"Max retry attempts ({self.max_attempts}) reached"
            logger.warning(reason)
            return False, reason

        # Check for retry loop (same error multiple times)
        if self._is_retry_loop(error):
            reason = "Retry loop detected (same error in multiple attempts)"
            logger.warning(reason)
            return False, reason

        # Check if state machine allows retry
        current_state = self.state_machine.get_current_state()
        if not self.state_machine.should_retry(self.current_attempt, self.max_attempts):
            reason = f"Current state ({current_state}) does not allow retry"
            logger.warning(reason)
            return False, reason

        # All checks passed - we should retry
        logger.info(f"Retry approved (attempt {self.current_attempt + 1}/{self.max_attempts})")
        return True, None

    def _is_retry_loop(self, current_error: str) -> bool:
        """
        Detect retry loop (same error in multiple attempts).

        Args:
            current_error: Current error message

        Returns:
            True if retry loop detected
        """
        if not self.error_history:
            return False

        # Check if current error is identical to previous errors
        # Simple heuristic: if first 100 chars match, consider it the same error
        current_error_key = current_error[:100].lower() if current_error else ""

        matching_errors = 0
        for prev_error in self.error_history:
            prev_error_key = prev_error[:100].lower() if prev_error else ""
            if current_error_key == prev_error_key:
                matching_errors += 1

        # If we've seen the same error twice already, it's a loop
        if matching_errors >= 2:
            logger.warning(f"Retry loop detected: error seen {matching_errors} times")
            return True

        return False

    def prepare_retry(
        self,
        error: str,
        failed_step: Optional[int] = None
    ) -> tuple[bool, Optional[str], Optional[int]]:
        """
        Prepare for retry attempt.

        Args:
            error: Error that caused failure
            failed_step: Step where failure occurred

        Returns:
            Tuple of (should_retry, checkpoint_uri_to_resume_from, delay_seconds)
        """
        # Record error
        self.error_history.append(error)

        # Check if we should retry
        should_retry, reason = self.should_retry(error)

        if not should_retry:
            logger.info(f"Will not retry: {reason}")
            return False, None, None

        # Get latest checkpoint for resume
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()

        if not latest_checkpoint:
            logger.warning("No checkpoint available for resume, will start from scratch")
            checkpoint_uri = None
        else:
            checkpoint_uri = latest_checkpoint['uri']
            checkpoint_step = latest_checkpoint['step']
            logger.info(f"Will resume from checkpoint at step {checkpoint_step}: {checkpoint_uri}")

        # Calculate retry delay
        attempt_index = min(self.current_attempt - 1, len(self.delay_minutes) - 1)
        delay_minutes = self.delay_minutes[attempt_index]
        delay_seconds = delay_minutes * 60

        logger.info(
            f"Retry scheduled: attempt {self.current_attempt + 1}/{self.max_attempts} "
            f"in {delay_minutes} minutes"
        )

        # Update state machine
        self.state_machine.transition_to(
            TrainingState.RETRYING,
            reason=f"Retry attempt {self.current_attempt + 1}",
            metadata={
                "error": error,
                "failed_step": failed_step,
                "checkpoint_to_resume": checkpoint_uri,
                "delay_minutes": delay_minutes
            }
        )

        # Record retry in MongoDB
        self.persistence.record_retry_attempt(
            attempt=self.current_attempt + 1,
            resumed_from=checkpoint_uri,
            error=error
        )

        return True, checkpoint_uri, delay_seconds

    def execute_retry(
        self,
        checkpoint_uri: Optional[str],
        delay_seconds: int
    ) -> Optional[str]:
        """
        Execute retry after delay.

        Args:
            checkpoint_uri: Checkpoint to resume from
            delay_seconds: Delay before retry

        Returns:
            Local path to checkpoint if resume needed, None if starting fresh
        """
        # Wait for delay
        logger.info(f"Waiting {delay_seconds}s before retry...")
        time.sleep(delay_seconds)

        # Increment attempt counter
        self.current_attempt += 1

        logger.info(f"Starting retry attempt {self.current_attempt}/{self.max_attempts}")

        # Download checkpoint if available
        if checkpoint_uri:
            local_checkpoint_dir = f"/workspace/resume/checkpoint-{self.current_attempt}"

            try:
                self.checkpoint_manager.download_checkpoint_for_resume(
                    checkpoint_uri,
                    local_checkpoint_dir
                )
                logger.info(f"✅ Checkpoint downloaded for resume: {local_checkpoint_dir}")
                return local_checkpoint_dir

            except Exception as e:
                logger.error(f"Failed to download checkpoint for resume: {e}")
                logger.warning("Will start training from scratch")
                return None

        return None

    def get_retry_status(self) -> Dict[str, Any]:
        """Get current retry status."""
        return {
            "current_attempt": self.current_attempt,
            "max_attempts": self.max_attempts,
            "can_retry": self.current_attempt < self.max_attempts,
            "errors_seen": len(self.error_history)
        }
