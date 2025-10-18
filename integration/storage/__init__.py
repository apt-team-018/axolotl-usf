"""Storage abstraction layer for multiple backends."""

from integration.storage.base import StorageBackend
from integration.storage.manager import StorageManager
from integration.storage.s3 import S3Backend
from integration.storage.azure import AzureBlobBackend
from integration.storage.gcs import GCSBackend
from integration.storage.filesystem import FilesystemBackend
from integration.storage.huggingface import HuggingFaceBackend

__all__ = [
    "StorageBackend",
    "StorageManager",
    "S3Backend",
    "AzureBlobBackend",
    "GCSBackend",
    "FilesystemBackend",
    "HuggingFaceBackend",
]
