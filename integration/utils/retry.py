"""
Retry utilities for handling transient failures.
Uses tenacity for exponential backoff and jittered retries.
"""

from functools import wraps
from typing import Callable, Optional, Type, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import logging

from integration.utils.logging_config import get_logger

logger = get_logger(__name__)


# Common retry configurations
STORAGE_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, IOError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)

NETWORK_RETRY = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)

API_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO)
)


def retry_on_exception(
    max_attempts: int = 3,
    backoff_multiplier: float = 1,
    min_wait: float = 1,
    max_wait: float = 60,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_multiplier: Multiplier for exponential backoff
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Example:
        @retry_on_exception(max_attempts=5, exceptions=(ConnectionError,))
        def download_file(url):
            # ... download logic
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


class RetryableOperation:
    """Context manager for retryable operations with custom logic."""

    def __init__(
        self,
        operation_name: str,
        max_attempts: int = 3,
        backoff_multiplier: float = 1,
        min_wait: float = 1,
        max_wait: float = 60
    ):
        """
        Initialize retryable operation.

        Args:
            operation_name: Name of the operation for logging
            max_attempts: Maximum retry attempts
            backoff_multiplier: Backoff multiplier
            min_wait: Minimum wait time
            max_wait: Maximum wait time
        """
        self.operation_name = operation_name
        self.max_attempts = max_attempts
        self.backoff_multiplier = backoff_multiplier
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.attempt = 0

    def __enter__(self):
        """Enter context manager."""
        self.attempt += 1
        logger.info(
            f"Starting {self.operation_name} (attempt {self.attempt}/{self.max_attempts})"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is None:
            logger.info(f"{self.operation_name} completed successfully")
            return True

        if self.attempt < self.max_attempts:
            wait_time = min(
                self.max_wait,
                self.min_wait * (self.backoff_multiplier ** (self.attempt - 1))
            )
            logger.warning(
                f"{self.operation_name} failed (attempt {self.attempt}), "
                f"retrying in {wait_time:.1f}s: {exc_val}"
            )
            import time
            time.sleep(wait_time)
            return False  # Will re-raise exception
        else:
            logger.error(
                f"{self.operation_name} failed after {self.max_attempts} attempts: {exc_val}"
            )
            return False  # Will re-raise exception


# Convenience functions for common operations
def with_storage_retry(func: Callable) -> Callable:
    """
    Decorator for storage operations with automatic retry.

    Example:
        @with_storage_retry
        def upload_file(self, path, uri):
            # ... upload logic
    """
    @wraps(func)
    @STORAGE_RETRY
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def with_network_retry(func: Callable) -> Callable:
    """
    Decorator for network operations with automatic retry.

    Example:
        @with_network_retry
        def download_model(url):
            # ... download logic
    """
    @wraps(func)
    @NETWORK_RETRY
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def with_api_retry(func: Callable) -> Callable:
    """
    Decorator for API calls with automatic retry.

    Example:
        @with_api_retry
        def call_api(endpoint):
            # ... API call logic
    """
    @wraps(func)
    @API_RETRY
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
