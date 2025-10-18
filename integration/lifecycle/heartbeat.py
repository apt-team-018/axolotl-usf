"""
Heartbeat monitoring for training crash detection.
Runs in background thread and detects when training process becomes unresponsive.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable

from integration.lifecycle.states import TrainingState
from integration.lifecycle.persistence import JobStatePersistence
from integration.utils.logging_config import get_logger

logger = get_logger(__name__)


class HeartbeatMonitor:
    """
    Background heartbeat monitor that detects training crashes.
    Updates MongoDB every N seconds and checks for timeout.
    """

    def __init__(
        self,
        persistence: JobStatePersistence,
        interval_seconds: int = 60,
        timeout_seconds: int = 3600,
        on_timeout_callback: Optional[Callable] = None
    ):
        """
        Initialize heartbeat monitor.

        Args:
            persistence: MongoDB persistence layer
            interval_seconds: How often to send heartbeat (default: 60s)
            timeout_seconds: Timeout threshold (default: 3600s = 1 hour)
            on_timeout_callback: Callback to call on timeout detection
        """
        self.persistence = persistence
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.on_timeout_callback = on_timeout_callback

        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.last_heartbeat_sent = datetime.utcnow()

        logger.info(
            f"Heartbeat monitor initialized: interval={interval_seconds}s, timeout={timeout_seconds}s"
        )

    def start(self) -> None:
        """Start heartbeat monitoring thread."""
        if self.is_running:
            logger.warning("Heartbeat monitor already running")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()

        logger.info("✅ Heartbeat monitor started")

    def stop(self) -> None:
        """Stop heartbeat monitoring."""
        if not self.is_running:
            return

        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)

        logger.info("Heartbeat monitor stopped")

    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop (runs in background thread)."""
        logger.debug("Heartbeat loop started")

        while self.is_running:
            try:
                # Send heartbeat
                self.persistence.update_heartbeat()
                self.last_heartbeat_sent = datetime.utcnow()

                logger.debug("Heartbeat sent to MongoDB")

                # Check for timeout (in case monitoring external job)
                if self.persistence.check_heartbeat_timeout():
                    logger.critical("⚠️ HEARTBEAT TIMEOUT DETECTED!")

                    # Update state to TIMEOUT
                    self.persistence.update_state(
                        TrainingState.TIMEOUT,
                        reason="No heartbeat for > 1 hour",
                        metadata={
                            "last_heartbeat": self.last_heartbeat_sent.isoformat(),
                            "timeout_seconds": self.timeout_seconds
                        }
                    )

                    # Call timeout callback if provided
                    if self.on_timeout_callback:
                        try:
                            self.on_timeout_callback()
                        except Exception as e:
                            logger.error(f"Timeout callback failed: {e}")

                    # Stop monitoring
                    self.is_running = False
                    break

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

            # Sleep for interval
            time.sleep(self.interval_seconds)

        logger.debug("Heartbeat loop ended")

    def send_immediate_heartbeat(self) -> None:
        """Send heartbeat immediately (outside normal interval)."""
        try:
            self.persistence.update_heartbeat()
            self.last_heartbeat_sent = datetime.utcnow()
            logger.debug("Immediate heartbeat sent")
        except Exception as e:
            logger.error(f"Failed to send immediate heartbeat: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
