"""Azure Blob Storage backend."""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import AzureError
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

from integration.storage.base import StorageBackend


class AzureBlobBackend(StorageBackend):
    """Azure Blob Storage backend implementation."""

    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        """
        Initialize Azure Blob backend.

        Args:
            account_name: Azure storage account name
            account_key: Azure storage account key
            connection_string: Azure storage connection string (alternative to name/key)
        """
        if not HAS_AZURE:
            raise ImportError(
                "azure-storage-blob is required for AzureBlobBackend. "
                "Install with: pip install azure-storage-blob"
            )

        if connection_string:
            self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        else:
            account_name = account_name or os.getenv('AZURE_STORAGE_ACCOUNT')
            account_key = account_key or os.getenv('AZURE_STORAGE_KEY')

            if not account_name or not account_key:
                raise ValueError("Azure account_name and account_key are required")

            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service = BlobServiceClient(
                account_url=account_url,
                credential=account_key
            )

    def _parse_azure_uri(self, uri: str) -> tuple[str, str]:
        """Parse Azure URI into container and blob name."""
        # Handle azure://container/blob or https://account.blob.core.windows.net/container/blob
        if uri.startswith('azure://'):
            parsed = urlparse(uri)
            container = parsed.netloc
            blob = parsed.path.lstrip('/')
        elif 'blob.core.windows.net' in uri:
            parsed = urlparse(uri)
            path_parts = parsed.path.lstrip('/').split('/', 1)
            container = path_parts[0]
            blob = path_parts[1] if len(path_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid Azure URI format: {uri}")

        return container, blob

    def download_dataset(self, uri: str, local_path: str) -> None:
        """Download dataset from Azure Blob to local path."""
        container, blob = self._parse_azure_uri(uri)
        local_file = self._ensure_local_dir(local_path)

        print(f"Downloading azure://{container}/{blob} to {local_file}")

        try:
            blob_client = self.blob_service.get_blob_client(container=container, blob=blob)
            with open(local_file, 'wb') as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            print(f"✅ Successfully downloaded dataset")
        except AzureError as e:
            raise RuntimeError(f"Failed to download from Azure Blob: {e}")

    def upload_checkpoint(self, local_path: str, remote_uri: str) -> None:
        """Upload checkpoint directory to Azure Blob."""
        import tarfile
        import tempfile

        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {local_path}")

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create tarball
            with tarfile.open(tmp_path, 'w:gz') as tar:
                tar.add(local_dir, arcname=local_dir.name)

            # Upload tarball
            container, blob = self._parse_azure_uri(remote_uri)
            if not blob.endswith('.tar.gz'):
                blob = f"{blob}.tar.gz"

            print(f"Uploading checkpoint to azure://{container}/{blob}")
            blob_client = self.blob_service.get_blob_client(container=container, blob=blob)
            with open(tmp_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"✅ Checkpoint uploaded successfully")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def upload_model(self, local_path: str, remote_uri: str) -> None:
        """Upload final model directory to Azure Blob."""
        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {local_path}")

        container, base_blob = self._parse_azure_uri(remote_uri)

        # Upload all files in the model directory
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                blob_name = f"{base_blob}/{relative_path}".replace('\\', '/')

                print(f"Uploading {file_path.name} to azure://{container}/{blob_name}")
                blob_client = self.blob_service.get_blob_client(container=container, blob=blob_name)
                with open(file_path, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)

        print(f"✅ Model uploaded successfully to azure://{container}/{base_blob}")

    def list_objects(self, uri: str) -> list[str]:
        """List blobs in Azure container with given prefix."""
        container, prefix = self._parse_azure_uri(uri)

        try:
            container_client = self.blob_service.get_container_client(container)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except AzureError as e:
            raise RuntimeError(f"Failed to list Azure blobs: {e}")

    def upload_file(self, local_path: str, remote_uri: str) -> None:
        """Upload a single file to Azure Blob."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        container, blob = self._parse_azure_uri(remote_uri)

        try:
            blob_client = self.blob_service.get_blob_client(container=container, blob=blob)
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"✅ Uploaded {local_path} to azure://{container}/{blob}")
        except AzureError as e:
            raise RuntimeError(f"Failed to upload file to Azure Blob: {e}")

    def download_file(self, uri: str, local_path: str) -> None:
        """Download a single file from Azure Blob."""
        container, blob = self._parse_azure_uri(uri)
        local_file = self._ensure_local_dir(local_path)

        try:
            blob_client = self.blob_service.get_blob_client(container=container, blob=blob)
            with open(local_file, 'wb') as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            print(f"✅ Downloaded azure://{container}/{blob} to {local_file}")
        except AzureError as e:
            raise RuntimeError(f"Failed to download file from Azure Blob: {e}")
