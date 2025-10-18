"""
Training state machine for lifecycle management.
Tracks all possible training states and valid transitions.
"""

from enum import Enum
from typing import Set, Optional, Dict, Any
from datetime import datetime
import threading

from integration.utils.logging_config import get_logger

logger = get_logger(__name__)


class TrainingState(str, Enum):
    """All possible training states."""

    # Initial states
    INITIALIZING = "INITIALIZING"       # Job just created
    VALIDATING = "VALIDATING"           # Validating configuration
    HEALTHY = "HEALTHY"                 # Health checks passed

    # Active states
    RUNNING = "RUNNING"                 # Training in progress
    PAUSED = "PAUSED"                   # Training paused (optional)

    # Terminal states (successful)
    SUCCESS = "SUCCESS"                 # Training completed successfully

    # Terminal states (failure)
    FAILED = "FAILED"                   # Training failed with error
    CRASHED = "CRASHED"                 # Process crashed
    TIMEOUT = "TIMEOUT"                 # No heartbeat for 1+ hour

    # Retry states
    RETRYING = "RETRYING"               # Preparing to retry
    TERMINAL_FAILURE = "TERMINAL_FAILURE"  # Failed after max retries

    # Cleanup states
    CLEANUP = "CLEANUP"                 # Uploading final artifacts
    COMPLETED = "COMPLETED"             # All operations complete


class TrainingStateMachine:
    """
    Thread-safe state machine for training lifecycle.
    Ensures only valid state transitions occur.
    """

    # Define valid state transitions
    VALID_TRANSITIONS: Dict[TrainingState, Set[TrainingState]] = {
        TrainingState.INITIALIZING: {
            TrainingState.VALIDATING,
            TrainingState.FAILED
        },
        TrainingState.VALIDATING: {
            TrainingState.HEALTHY,
            TrainingState.FAILED
        },
        TrainingState.HEALTHY: {
            TrainingState.RUNNING,
            TrainingState.FAILED
        },
        TrainingState.RUNNING: {
            TrainingState.SUCCESS,
            TrainingState.FAILED,
            TrainingState.CRASHED,
            TrainingState.TIMEOUT,
            TrainingState.PAUSED
        },
        TrainingState.PAUSED: {
            TrainingState.RUNNING,
            TrainingState.FAILED
        },
        TrainingState.FAILED: {
            TrainingState.RETRYING,
            TrainingState.TERMINAL_FAILURE,
            TrainingState.CLEANUP
        },
        TrainingState.CRASHED: {
            TrainingState.RETRYING,
            TrainingState.TERMINAL_FAILURE,
            TrainingState.CLEANUP
        },
        TrainingState.TIMEOUT: {
            TrainingState.RETRYING,
            TrainingState.TERMINAL_FAILURE,
            TrainingState.CLEANUP
        },
        TrainingState.RETRYING: {
            TrainingState.RUNNING,
            TrainingState.TERMINAL_FAILURE
        },
        TrainingState.SUCCESS: {
            TrainingState.CLEANUP
        },
        TrainingState.TERMINAL_FAILURE: {
            TrainingState.CLEANUP
        },
        TrainingState.CLEANUP: {
            TrainingState.COMPLETED
        },
        TrainingState.COMPLETED: set()  # Final state, no transitions
    }

    def __init__(self, job_id: str, initial_state: TrainingState = TrainingState.INITIALIZING):
        """
        Initialize state machine.

        Args:
            job_id: Job identifier
            initial_state: Initial state (default: INITIALIZING)
        """
        self.job_id = job_id
        self.current_state = initial_state
        self.state_history: list[Dict[str, Any]] = []
        self.lock = threading.RLock()  # Thread-safe state updates

        # Record initial state
        self._record_transition(None, initial_state, "Initialized")

        logger.info(f"State machine initialized for job {job_id}: {initial_state.value}")

    def transition_to(
        self,
        new_state: TrainingState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to transition to a new state.

        Args:
            new_state: Target state
            reason: Reason for transition
            metadata: Additional metadata

        Returns:
            True if transition successful, False if invalid
        """
        with self.lock:
            if not self._is_valid_transition(self.current_state, new_state):
                logger.error(
                    f"Invalid state transition: {self.current_state.value} -> {new_state.value}",
                    extra={"job_id": self.job_id}
                )
                return False

            old_state = self.current_state
            self.current_state = new_state

            self._record_transition(old_state, new_state, reason, metadata)

            logger.info(
                f"State transition: {old_state.value} -> {new_state.value}",
                extra={
                    "job_id": self.job_id,
                    "reason": reason,
                    "old_state": old_state.value,
                    "new_state": new_state.value
                }
            )

            return True

    def _is_valid_transition(self, from_state: TrainingState, to_state: TrainingState) -> bool:
        """Check if state transition is valid."""
        return to_state in self.VALID_TRANSITIONS.get(from_state, set())

    def _record_transition(
        self,
        from_state: Optional[TrainingState],
        to_state: TrainingState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record state transition in history."""
        record = {
            "timestamp": datetime.utcnow(),
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value,
            "reason": reason,
            "metadata": metadata or {}
        }
        self.state_history.append(record)

    def get_current_state(self) -> TrainingState:
        """Get current state (thread-safe)."""
        with self.lock:
            return self.current_state

    def is_terminal_state(self) -> bool:
        """Check if current state is terminal."""
        terminal_states = {
            TrainingState.SUCCESS,
            TrainingState.TERMINAL_FAILURE,
            TrainingState.COMPLETED
        }
        with self.lock:
            return self.current_state in terminal_states

    def is_failure_state(self) -> bool:
        """Check if current state represents a failure."""
        failure_states = {
            TrainingState.FAILED,
            TrainingState.CRASHED,
            TrainingState.TIMEOUT,
            TrainingState.TERMINAL_FAILURE
        }
        with self.lock:
            return self.current_state in failure_states

    def should_retry(self, current_attempt: int, max_attempts: int) -> bool:
        """
        Check if job should be retried based on current state.

        Args:
            current_attempt: Current retry attempt number
            max_attempts: Maximum allowed attempts

        Returns:
            True if should retry, False otherwise
        """
        with self.lock:
            if self.current_state not in {TrainingState.FAILED, TrainingState.CRASHED, TrainingState.TIMEOUT}:
                return False

            return current_attempt < max_attempts

    def get_state_history(self) -> list[Dict[str, Any]]:
        """Get complete state transition history."""
        with self.lock:
            return self.state_history.copy()
