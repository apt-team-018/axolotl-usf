"""HuggingFace Hub storage backend."""

import os
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, upload_folder, upload_file
    from huggingface_hub.utils import HfHubHTTPError
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

try:
    from datasets import load_dataset, DatasetDict
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from integration.storage.base import StorageBackend


class HuggingFaceBackend(StorageBackend):
    """HuggingFace Hub storage backend implementation."""

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize HuggingFace backend.

        Args:
            hf_token: HuggingFace API token
        """
        if not HAS_HF_HUB:
            raise ImportError(
                "huggingface_hub is required for HuggingFaceBackend. "
                "Install with: pip install huggingface_hub"
            )

        self.token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
        self.api = HfApi(token=self.token)

    def _parse_hf_uri(self, uri: str) -> tuple[str, Optional[str]]:
        """Parse HuggingFace URI into repo_id and optional path."""
        # Handle hf://repo_id/path or huggingface://repo_id/path
        if uri.startswith('hf://'):
            parts = uri[5:].split('/', 1)
        elif uri.startswith('huggingface://'):
            parts = uri[14:].split('/', 1)
        else:
            # Assume it's just a repo_id
            parts = [uri]

        repo_id = parts[0]
        path = parts[1] if len(parts) > 1 else None

        return repo_id, path

    def download_dataset(self, uri: str, local_path: str) -> None:
        """Download dataset from HuggingFace Hub to local path."""
        repo_id, dataset_name = self._parse_hf_uri(uri)
        local_dir = self._ensure_local_dir(local_path)

        print(f"Downloading HuggingFace dataset: {repo_id}")

        try:
            # Try loading as a dataset first
            if HAS_DATASETS:
                try:
                    dataset = load_dataset(repo_id, token=self.token)
                    if isinstance(dataset, DatasetDict):
                        # Save all splits
                        dataset.save_to_disk(str(local_dir))
                    else:
                        # Save single split
                        dataset.save_to_disk(str(local_dir))
                    print(f"✅ Successfully downloaded dataset")
                    return
                except Exception as e:
                    print(f"Not a dataset, trying as model/file: {e}")

            # Fall back to downloading as files
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=self.token,
                repo_type="dataset"
            )
            print(f"✅ Successfully downloaded files")

        except HfHubHTTPError as e:
            raise RuntimeError(f"Failed to download from HuggingFace Hub: {e}")

    def upload_checkpoint(self, local_path: str, remote_uri: str) -> None:
        """Upload checkpoint directory to HuggingFace Hub."""
        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {local_path}")

        repo_id, path_in_repo = self._parse_hf_uri(remote_uri)

        print(f"Uploading checkpoint to HuggingFace: {repo_id}")

        try:
            # Create repo if it doesn't exist
            try:
                self.api.create_repo(repo_id, repo_type="model", exist_ok=True)
            except Exception:
                pass  # Repo might already exist

            # Upload folder
            upload_folder(
                folder_path=str(local_dir),
                repo_id=repo_id,
                path_in_repo=path_in_repo or "checkpoints",
                token=self.token,
                repo_type="model"
            )
            print(f"✅ Checkpoint uploaded successfully")

        except HfHubHTTPError as e:
            raise RuntimeError(f"Failed to upload checkpoint to HuggingFace Hub: {e}")

    def upload_model(self, local_path: str, remote_uri: str) -> None:
        """Upload final model directory to HuggingFace Hub."""
        local_dir = Path(local_path)
        if not local_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {local_path}")

        repo_id, _ = self._parse_hf_uri(remote_uri)

        print(f"Uploading model to HuggingFace: {repo_id}")

        try:
            # Create repo if it doesn't exist
            try:
                self.api.create_repo(repo_id, repo_type="model", exist_ok=True)
            except Exception:
                pass  # Repo might already exist

            # Upload entire model directory
            upload_folder(
                folder_path=str(local_dir),
                repo_id=repo_id,
                token=self.token,
                repo_type="model"
            )
            print(f"✅ Model uploaded successfully to {repo_id}")

        except HfHubHTTPError as e:
            raise RuntimeError(f"Failed to upload model to HuggingFace Hub: {e}")

    def list_objects(self, uri: str) -> list[str]:
        """List files in HuggingFace repository."""
        repo_id, path = self._parse_hf_uri(uri)

        try:
            files = self.api.list_repo_files(repo_id=repo_id, token=self.token)
            if path:
                # Filter by path prefix
                files = [f for f in files if f.startswith(path)]
            return files
        except HfHubHTTPError as e:
            raise RuntimeError(f"Failed to list HuggingFace repo files: {e}")

    def upload_file(self, local_path: str, remote_uri: str) -> None:
        """Upload a single file to HuggingFace Hub."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        repo_id, path_in_repo = self._parse_hf_uri(remote_uri)

        if not path_in_repo:
            path_in_repo = os.path.basename(local_path)

        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=self.token,
                repo_type="model"
            )
            print(f"✅ Uploaded {local_path} to {repo_id}/{path_in_repo}")
        except HfHubHTTPError as e:
            raise RuntimeError(f"Failed to upload file to HuggingFace Hub: {e}")

    def download_file(self, uri: str, local_path: str) -> None:
        """Download a single file from HuggingFace Hub."""
        repo_id, filename = self._parse_hf_uri(uri)
        local_file = self._ensure_local_dir(local_path)

        if not filename:
            raise ValueError("Filename required in URI for download_file")

        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=self.token,
                local_dir=str(local_file.parent)
            )
            # Move to desired location if different
            if downloaded_path != str(local_file):
                import shutil
                shutil.move(downloaded_path, local_file)
            print(f"✅ Downloaded {repo_id}/{filename} to {local_file}")
        except HfHubHTTPError as e:
            raise RuntimeError(f"Failed to download file from HuggingFace Hub: {e}")
