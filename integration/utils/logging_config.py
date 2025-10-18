"""
Structured logging configuration for production use.
Provides consistent logging across all components with proper levels and formatting.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'job_id'):
            log_data['job_id'] = record.job_id
        if hasattr(record, 'node_rank'):
            log_data['node_rank'] = record.node_rank
        if hasattr(record, 'gpu_id'):
            log_data['gpu_id'] = record.gpu_id

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add any custom extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info', 'job_id', 'node_rank', 'gpu_id']:
                log_data[key] = value

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for human-readable output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥'
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and icons."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        icon = self.ICONS.get(record.levelname, '')
        reset = self.COLORS['RESET']

        # Format: [TIME] ICON LEVEL - MESSAGE
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        formatted = f"{color}[{timestamp}] {icon} {record.levelname}{reset} - {record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = False,
    job_id: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_logs: Use JSON formatting (for production)
        job_id: Job ID to include in logs

    Returns:
        Configured root logger
    """
    # Get log level from environment or parameter
    log_level_str = os.getenv('LOG_LEVEL', log_level).upper()
    log_level_int = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_int)

    if json_logs or os.getenv('JSON_LOGS', '').lower() == 'true':
        # Production: JSON logs
        console_handler.setFormatter(StructuredFormatter())
    else:
        # Development: Colored logs
        console_handler.setFormatter(ColoredConsoleFormatter())

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file or os.getenv('LOG_FILE'):
        log_file_path = Path(log_file or os.getenv('LOG_FILE'))
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level_int)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Set up library loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str, job_id: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with optional job context.

    Args:
        name: Logger name (usually __name__)
        job_id: Optional job ID for context

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # Add job_id to all logs from this logger if provided
    if job_id:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.job_id = job_id
            record.node_rank = int(os.getenv('NODE_RANK', os.getenv('RANK', 0)))
            record.gpu_id = int(os.getenv('LOCAL_RANK', 0))
            return record

        logging.setLogRecordFactory(record_factory)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Adapter to add contextual information to all log messages."""

    def process(self, msg, kwargs):
        """Add extra context to log message."""
        # Add context from self.extra
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        return msg, kwargs


def get_contextual_logger(
    name: str,
    job_id: Optional[str] = None,
    **context
) -> LoggerAdapter:
    """
    Get a logger with persistent context.

    Args:
        name: Logger name
        job_id: Job ID
        **context: Additional context to include in all logs

    Returns:
        LoggerAdapter with context
    """
    logger = get_logger(name, job_id)

    extra = context.copy()
    if job_id:
        extra['job_id'] = job_id
    extra['node_rank'] = int(os.getenv('NODE_RANK', os.getenv('RANK', 0)))
    extra['gpu_id'] = int(os.getenv('LOCAL_RANK', 0))

    return LoggerAdapter(logger, extra)
