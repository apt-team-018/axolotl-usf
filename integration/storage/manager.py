"""Storage manager factory for all backend types."""

from typing import Dict, Type, Any, Optional
from integration.storage.base import StorageBackend
from integration.storage.s3 import S3Backend
from integration.storage.azure import AzureBlobBackend
from integration.storage.gcs import GCSBackend
from integration.storage.filesystem import FilesystemBackend
from integration.storage.huggingface import HuggingFaceBackend


class StorageManager:
    """Factory for creating storage backend instances."""

    BACKENDS: Dict[str, Type[StorageBackend]] = {
        's3': S3Backend,
        'azure': AzureBlobBackend,
        'gcs': GCSBackend,
        'fs': FilesystemBackend,
        'filesystem': FilesystemBackend,
        'hf': HuggingFaceBackend,
        'huggingface': HuggingFaceBackend,
    }

    @classmethod
    def get_backend(cls, storage_type: str, **kwargs: Any) -> StorageBackend:
        """
        Get a storage backend instance.

        Args:
            storage_type: Type of storage backend (s3, azure, gcs, fs, hf)
            **kwargs: Backend-specific initialization parameters

        Returns:
            Initialized storage backend instance

        Raises:
            ValueError: If storage type is unknown

        Examples:
            >>> # S3 backend
            >>> backend = StorageManager.get_backend(
            ...     's3',
            ...     aws_access_key_id='...',
            ...     aws_secret_access_key='...',
            ...     region='us-east-1'
            ... )

            >>> # Azure backend
            >>> backend = StorageManager.get_backend(
            ...     'azure',
            ...     account_name='...',
            ...     account_key='...'
            ... )

            >>> # HuggingFace backend
            >>> backend = StorageManager.get_backend(
            ...     'hf',
            ...     hf_token='...'
            ... )
        """
        storage_type_lower = storage_type.lower()

        backend_class = cls.BACKENDS.get(storage_type_lower)
        if not backend_class:
            raise ValueError(
                f"Unknown storage type: {storage_type}. "
                f"Supported types: {', '.join(cls.BACKENDS.keys())}"
            )

        return backend_class(**kwargs)

    @classmethod
    def auto_detect_backend(cls, uri: str, **kwargs: Any) -> StorageBackend:
        """
        Auto-detect storage backend from URI and create instance.

        Args:
            uri: Storage URI (e.g., s3://bucket/path, azure://container/blob)
            **kwargs: Backend-specific initialization parameters

        Returns:
            Initialized storage backend instance

        Examples:
            >>> backend = StorageManager.auto_detect_backend('s3://my-bucket/data.json')
            >>> backend = StorageManager.auto_detect_backend('azure://container/blob')
            >>> backend = StorageManager.auto_detect_backend('hf://username/dataset')
        """
        uri_lower = uri.lower()

        if uri_lower.startswith('s3://'):
            return cls.get_backend('s3', **kwargs)
        elif uri_lower.startswith('azure://') or 'blob.core.windows.net' in uri_lower:
            return cls.get_backend('azure', **kwargs)
        elif uri_lower.startswith('gs://') or uri_lower.startswith('gcs://'):
            return cls.get_backend('gcs', **kwargs)
        elif uri_lower.startswith('hf://') or uri_lower.startswith('huggingface://'):
            return cls.get_backend('hf', **kwargs)
        elif uri_lower.startswith('file://') or uri_lower.startswith('fs://') or uri_lower.startswith('/'):
            return cls.get_backend('fs', **kwargs)
        else:
            # Default to filesystem for relative paths
            return cls.get_backend('fs', **kwargs)

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[StorageBackend]) -> None:
        """
        Register a custom storage backend.

        Args:
            name: Name to register the backend under
            backend_class: Backend class (must inherit from StorageBackend)
        """
        if not issubclass(backend_class, StorageBackend):
            raise TypeError("Backend class must inherit from StorageBackend")

        cls.BACKENDS[name.lower()] = backend_class
