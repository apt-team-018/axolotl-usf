"""Base storage backend interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class StorageBackend(ABC):
    """Abstract base class for all storage backends."""

    @abstractmethod
    def download_dataset(self, uri: str, local_path: str) -> None:
        """
        Download dataset from remote storage to local path.

        Args:
            uri: Remote URI of the dataset
            local_path: Local path to save the dataset
        """
        pass

    @abstractmethod
    def upload_checkpoint(self, local_path: str, remote_uri: str) -> None:
        """
        Upload checkpoint from local path to remote storage.

        Args:
            local_path: Local path of the checkpoint
            remote_uri: Remote URI to upload to
        """
        pass

    @abstractmethod
    def upload_model(self, local_path: str, remote_uri: str) -> None:
        """
        Upload final model from local path to remote storage.

        Args:
            local_path: Local path of the model
            remote_uri: Remote URI to upload to
        """
        pass

    @abstractmethod
    def list_objects(self, uri: str) -> list[str]:
        """
        List objects at the given URI.

        Args:
            uri: Remote URI to list objects from

        Returns:
            List of object names/keys
        """
        pass

    @abstractmethod
    def upload_file(self, local_path: str, remote_uri: str) -> None:
        """
        Upload a single file to remote storage.

        Args:
            local_path: Local file path
            remote_uri: Remote URI to upload to
        """
        pass

    @abstractmethod
    def download_file(self, uri: str, local_path: str) -> None:
        """
        Download a single file from remote storage.

        Args:
            uri: Remote URI of the file
            local_path: Local path to save the file
        """
        pass

    def _ensure_local_dir(self, local_path: str) -> Path:
        """Ensure local directory exists."""
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
