"""Local filesystem and shared filesystem storage backend."""

import os
import shutil
from pathlib import Path
from typing import Optional

from integration.storage.base import StorageBackend


class FilesystemBackend(StorageBackend):
    """Local and shared filesystem storage backend."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize filesystem backend.

        Args:
            base_path: Base path for relative URIs (optional)
        """
        self.base_path = Path(base_path) if base_path else None

    def _resolve_path(self, uri: str) -> Path:
        """Resolve filesystem path from URI."""
        # Handle file:// URIs or plain paths
        if uri.startswith('file://'):
            path = Path(uri[7:])
        elif uri.startswith('fs://'):
            path = Path(uri[5:])
        else:
            path = Path(uri)

        # Make absolute if base_path is set
        if self.base_path and not path.is_absolute():
            path = self.base_path / path

        return path.resolve()

    def download_dataset(self, uri: str, local_path: str) -> None:
        """Copy dataset from source path to local path."""
        source = self._resolve_path(uri)
        dest = self._ensure_local_dir(local_path)

        if not source.exists():
            raise FileNotFoundError(f"Source dataset not found: {source}")

        print(f"Copying {source} to {dest}")

        if source.is_file():
            shutil.copy2(source, dest)
        else:
            shutil.copytree(source, dest, dirs_exist_ok=True)

        print(f"✅ Successfully copied dataset")

    def upload_checkpoint(self, local_path: str, remote_uri: str) -> None:
        """Copy checkpoint directory to remote path."""
        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {local_path}")

        remote_path = self._resolve_path(remote_uri)

        print(f"Copying checkpoint to {remote_path}")

        # Remove existing if present
        if remote_path.exists():
            if remote_path.is_dir():
                shutil.rmtree(remote_path)
            else:
                remote_path.unlink()

        # Copy directory
        shutil.copytree(local_dir, remote_path)
        print(f"✅ Checkpoint copied successfully")

    def upload_model(self, local_path: str, remote_uri: str) -> None:
        """Copy model directory to remote path."""
        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {local_path}")

        remote_path = self._resolve_path(remote_uri)

        print(f"Copying model to {remote_path}")

        # Remove existing if present
        if remote_path.exists():
            if remote_path.is_dir():
                shutil.rmtree(remote_path)
            else:
                remote_path.unlink()

        # Copy directory
        shutil.copytree(local_dir, remote_path)
        print(f"✅ Model copied successfully to {remote_path}")

    def list_objects(self, uri: str) -> list[str]:
        """List files in directory."""
        path = self._resolve_path(uri)

        if not path.exists():
            return []

        if path.is_file():
            return [str(path)]

        return [str(p) for p in path.rglob('*') if p.is_file()]

    def upload_file(self, local_path: str, remote_uri: str) -> None:
        """Copy a single file to remote path."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        remote_path = self._resolve_path(remote_uri)
        remote_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(local_path, remote_path)
        print(f"✅ Copied {local_path} to {remote_path}")

    def download_file(self, uri: str, local_path: str) -> None:
        """Copy a single file from source path."""
        source = self._resolve_path(uri)
        dest = self._ensure_local_dir(local_path)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        shutil.copy2(source, dest)
        print(f"✅ Copied {source} to {dest}")
