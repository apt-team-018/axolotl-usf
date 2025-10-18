"""Google Cloud Storage backend."""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

try:
    from google.cloud import storage
    from google.cloud.exceptions import GoogleCloudError
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

from integration.storage.base import StorageBackend


class GCSBackend(StorageBackend):
    """Google Cloud Storage backend implementation."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize GCS backend.

        Args:
            project_id: GCP project ID
            credentials_path: Path to GCP credentials JSON file
        """
        if not HAS_GCS:
            raise ImportError(
                "google-cloud-storage is required for GCSBackend. "
                "Install with: pip install google-cloud-storage"
            )

        # Set credentials if provided
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        self.client = storage.Client(project=project_id)

    def _parse_gcs_uri(self, uri: str) -> tuple[str, str]:
        """Parse GCS URI into bucket and blob name."""
        # Handle gs://bucket/blob or https://storage.googleapis.com/bucket/blob
        if uri.startswith('gs://'):
            parsed = urlparse(uri)
            bucket = parsed.netloc
            blob = parsed.path.lstrip('/')
        elif uri.startswith('gcs://'):
            parsed = urlparse(uri)
            bucket = parsed.netloc
            blob = parsed.path.lstrip('/')
        elif 'storage.googleapis.com' in uri:
            parsed = urlparse(uri)
            path_parts = parsed.path.lstrip('/').split('/', 1)
            bucket = path_parts[0]
            blob = path_parts[1] if len(path_parts) > 1 else ''
        else:
            raise ValueError(f"Invalid GCS URI format: {uri}")

        return bucket, blob

    def download_dataset(self, uri: str, local_path: str) -> None:
        """Download dataset from GCS to local path."""
        bucket_name, blob_name = self._parse_gcs_uri(uri)
        local_file = self._ensure_local_dir(local_path)

        print(f"Downloading gs://{bucket_name}/{blob_name} to {local_file}")

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(local_file))
            print(f"✅ Successfully downloaded dataset")
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to download from GCS: {e}")

    def upload_checkpoint(self, local_path: str, remote_uri: str) -> None:
        """Upload checkpoint directory to GCS."""
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
            bucket_name, blob_name = self._parse_gcs_uri(remote_uri)
            if not blob_name.endswith('.tar.gz'):
                blob_name = f"{blob_name}.tar.gz"

            print(f"Uploading checkpoint to gs://{bucket_name}/{blob_name}")
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(tmp_path)
            print(f"✅ Checkpoint uploaded successfully")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def upload_model(self, local_path: str, remote_uri: str) -> None:
        """Upload final model directory to GCS."""
        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {local_path}")

        bucket_name, base_blob = self._parse_gcs_uri(remote_uri)
        bucket = self.client.bucket(bucket_name)

        # Upload all files in the model directory
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                blob_name = f"{base_blob}/{relative_path}".replace('\\', '/')

                print(f"Uploading {file_path.name} to gs://{bucket_name}/{blob_name}")
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))

        print(f"✅ Model uploaded successfully to gs://{bucket_name}/{base_blob}")

    def list_objects(self, uri: str) -> list[str]:
        """List blobs in GCS bucket with given prefix."""
        bucket_name, prefix = self._parse_gcs_uri(uri)

        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to list GCS objects: {e}")

    def upload_file(self, local_path: str, remote_uri: str) -> None:
        """Upload a single file to GCS."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        bucket_name, blob_name = self._parse_gcs_uri(remote_uri)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded {local_path} to gs://{bucket_name}/{blob_name}")
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to upload file to GCS: {e}")

    def download_file(self, uri: str, local_path: str) -> None:
        """Download a single file from GCS."""
        bucket_name, blob_name = self._parse_gcs_uri(uri)
        local_file = self._ensure_local_dir(local_path)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(local_file))
            print(f"✅ Downloaded gs://{bucket_name}/{blob_name} to {local_file}")
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to download file from GCS: {e}")
