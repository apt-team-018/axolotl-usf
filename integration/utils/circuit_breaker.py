"""
Circuit Breaker - Production-grade failure handling.
Prevents cascading failures and enables graceful degradation.
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    Prevents cascading failures by failing fast when service is down.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "circuit"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to catch
            name: Circuit breaker name (for logging)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

        self._lock = threading.RLock()

        logger.info(f"Circuit breaker '{name}' initialized "
                   f"(threshold={failure_threshold}, timeout={recovery_timeout}s)")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit '{self.name}' half-open, testing recovery")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Will retry after {self.recovery_timeout}s"
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to test recovery"""
        if self.last_failure_time is None:
            return True

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit '{self.name}' recovered, closing circuit")

            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    logger.error(
                        f"Circuit '{self.name}' OPENED after {self.failure_count} failures"
                    )
                    self.state = CircuitState.OPEN
            else:
                logger.warning(
                    f"Circuit '{self.name}' failure {self.failure_count}/{self.failure_threshold}"
                )

    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            logger.info(f"Circuit '{self.name}' manually reset")
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitState.CLOSED

    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'threshold': self.failure_threshold,
                'last_failure': self.last_failure_time
            }


def circuit(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    name: Optional[str] = None
):
    """
    Decorator to add circuit breaker protection to a function.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying again
        expected_exception: Exception type to catch
        name: Circuit breaker name (defaults to function name)

    Example:
        @circuit(failure_threshold=3, recovery_timeout=30)
        def call_external_service():
            response = requests.get("https://api.example.com")
            return response.json()
    """
    def decorator(func):
        circuit_name = name or func.__name__
        cb = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=circuit_name
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)

        # Attach circuit breaker to function for inspection
        wrapper._circuit_breaker = cb

        return wrapper

    return decorator


class CircuitBreakerRegistry:
    """
    Registry to manage multiple circuit breakers.
    Useful for monitoring and administration.
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker"""
        with self._lock:
            self._breakers[breaker.name] = breaker
            logger.debug(f"Registered circuit breaker: {breaker.name}")

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        with self._lock:
            return self._breakers.get(name)

    def get_all_states(self) -> dict:
        """Get states of all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_state()
                for name, breaker in self._breakers.items()
            }

    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")


# Global registry
_registry = CircuitBreakerRegistry()


def get_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry"""
    return _registry
