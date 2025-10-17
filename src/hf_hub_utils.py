"""
Hugging Face Hub utilities for automatic checkpoint uploading
"""
import os
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, login, create_repo
import torch


class HFCheckpointUploader:
    """
    Automatically upload checkpoints to Hugging Face Hub during training

    Usage:
        uploader = HFCheckpointUploader(
            repo_id="your-username/deep-enhancer-checkpoints",
            token="hf_xxxxx"  # Optional if already logged in
        )

        # After saving checkpoint
        uploader.upload_checkpoint("./checkpoints/best.ckpt", commit_message="Epoch 10")
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
        auto_create_repo: bool = True
    ):
        """
        Initialize Hugging Face Hub uploader

        Args:
            repo_id: Repository ID in format "username/repo-name"
            token: HF token (optional if already logged in via `huggingface-cli login`)
            private: Whether to create a private repository
            auto_create_repo: Automatically create repo if it doesn't exist
        """
        self.repo_id = repo_id
        self.private = private

        # Login if token provided
        if token:
            login(token=token, add_to_git_credential=True)

        # Initialize HF API
        self.api = HfApi()

        # Create repo if needed
        if auto_create_repo:
            try:
                create_repo(
                    repo_id=repo_id,
                    private=private,
                    exist_ok=True,
                    repo_type="model"
                )
                print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
            except Exception as e:
                print(f"⚠ Warning: Could not create/verify repo ({e})")

    def upload_checkpoint(
        self,
        checkpoint_path: str,
        commit_message: Optional[str] = None,
        subfolder: str = "checkpoints"
    ):
        """
        Upload a single checkpoint file to HF Hub

        Args:
            checkpoint_path: Local path to checkpoint file
            commit_message: Commit message (default: auto-generated)
            subfolder: Subfolder in repo to upload to
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"⚠ Warning: Checkpoint not found: {checkpoint_path}")
            return

        # Auto-generate commit message if not provided
        if commit_message is None:
            commit_message = f"Upload {checkpoint_path.name}"

        try:
            print(f"📤 Uploading {checkpoint_path.name} to HF Hub...")

            self.api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=f"{subfolder}/{checkpoint_path.name}",
                repo_id=self.repo_id,
                commit_message=commit_message,
            )

            print(f"✓ Uploaded: https://huggingface.co/{self.repo_id}/tree/main/{subfolder}/{checkpoint_path.name}")

        except Exception as e:
            print(f"⚠ Failed to upload checkpoint: {e}")

    def upload_folder(
        self,
        folder_path: str,
        commit_message: Optional[str] = None,
        ignore_patterns: Optional[list] = None
    ):
        """
        Upload entire checkpoint folder to HF Hub

        Args:
            folder_path: Local folder path
            commit_message: Commit message
            ignore_patterns: List of patterns to ignore (e.g., ["*.tmp", "*.log"])
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"⚠ Warning: Folder not found: {folder_path}")
            return

        if commit_message is None:
            commit_message = f"Upload checkpoints from {folder_path.name}"

        try:
            print(f"📤 Uploading folder {folder_path} to HF Hub...")

            self.api.upload_folder(
                folder_path=str(folder_path),
                repo_id=self.repo_id,
                commit_message=commit_message,
                ignore_patterns=ignore_patterns or ["*.tmp", "*.log", "__pycache__"]
            )

            print(f"✓ Folder uploaded: https://huggingface.co/{self.repo_id}")

        except Exception as e:
            print(f"⚠ Failed to upload folder: {e}")

    def download_checkpoint(
        self,
        checkpoint_name: str,
        local_dir: str = "./checkpoints",
        subfolder: str = "checkpoints"
    ):
        """
        Download a checkpoint from HF Hub

        Args:
            checkpoint_name: Name of checkpoint file
            local_dir: Local directory to save to
            subfolder: Subfolder in repo to download from
        """
        try:
            from huggingface_hub import hf_hub_download

            print(f"📥 Downloading {checkpoint_name} from HF Hub...")

            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"{subfolder}/{checkpoint_name}",
                local_dir=local_dir,
            )

            print(f"✓ Downloaded to: {local_path}")
            return local_path

        except Exception as e:
            print(f"⚠ Failed to download checkpoint: {e}")
            return None


# Convenience function for quick setup
def create_uploader_from_env():
    """
    Create uploader from environment variables

    Set these environment variables:
        HF_REPO_ID="username/repo-name"
        HF_TOKEN="hf_xxxxx"  (optional if already logged in)
    """
    repo_id = os.getenv("HF_REPO_ID")
    token = os.getenv("HF_TOKEN")

    if not repo_id:
        raise ValueError(
            "HF_REPO_ID environment variable not set. "
            "Please set it to your HuggingFace repo (e.g., 'username/deep-enhancer-checkpoints')"
        )

    return HFCheckpointUploader(repo_id=repo_id, token=token)
