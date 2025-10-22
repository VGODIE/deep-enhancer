from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import os
from dotenv import load_dotenv

def upload_folder_to_hf(local_folder_path, hf_repo_id, hf_repo_type):
    load_dotenv()
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_large_folder(
        folder_path=local_folder_path,
        repo_id=hf_repo_id,
        repo_type=hf_repo_type,
    )

def upload_file_to_hf(local_file_path, hf_repo_id, hf_repo_type, remote_filpath):
    load_dotenv()
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=remote_filpath,
        repo_id=hf_repo_id,
        repo_type=hf_repo_type
    )

def download_file_from_hf(filename, hf_repo_id, hf_repo_type, local_dir=None):
    """Download a specific file from a Hugging Face Hub repository.

    Args:
        filename: Name of the file to download from the repo
        hf_repo_id: Repository ID (e.g., 'username/repo-name')
        hf_repo_type: Type of repo ('dataset', 'model', or 'space')
        local_dir: Local directory to save the file (default: current directory)

    Returns:
        str: Path to the downloaded file
    """
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    file_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=filename,
        repo_type=hf_repo_type,
        token=token,
        local_dir=local_dir
    )
    print(f"Downloaded file to: {file_path}")
    return file_path

def download_repo_from_hf(hf_repo_id, hf_repo_type, local_dir=None):
    """Download entire repository from Hugging Face Hub.

    Args:
        hf_repo_id: Repository ID (e.g., 'username/repo-name')
        hf_repo_type: Type of repo ('dataset', 'model', or 'space')
        local_dir: Local directory to save files (default: cache directory)

    Returns:
        str: Path to the downloaded repository
    """
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    repo_path = snapshot_download(
        repo_id=hf_repo_id,
        repo_type=hf_repo_type,
        token=token,
        local_dir=local_dir
    )
    print(f"Downloaded repository to: {repo_path}")
    return repo_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Upload or download files/repos from Hugging Face Hub')
    parser.add_argument('--action', type=str, required=True,
                        choices=['upload-file', 'upload-folder', 'download-file', 'download-repo'],
                        help='Action to perform')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='Hugging Face repository ID (e.g., username/repo-name)')
    parser.add_argument('--repo-type', type=str, default='dataset',
                        choices=['dataset', 'model', 'space'],
                        help='Type of repository')
    parser.add_argument('--local-path', type=str,
                        help='Local file/folder path (for uploads) or local directory (for downloads)')
    parser.add_argument('--remote-filepath', type=str,
                        help='Filename to download from the repository (for download-file action)')
    args = parser.parse_args()

    if args.action == 'upload-folder':
        if not args.local_path:
            parser.error('--local-path is required for upload-folder')
        upload_folder_to_hf(args.local_path, args.repo_id, args.repo_type)
    elif args.action == 'upload-file':
        if not args.local_path:
            parser.error('--local-path is required for upload-file')
        upload_file_to_hf(args.local_path, args.repo_id, args.repo_type, args.remote_filepath)
    elif args.action == 'download-file':
        if not args.filename:
            parser.error('--filename is required for download-file')
        download_file_from_hf(args.remote_filepath, args.repo_id, args.repo_type, args.local_path)
    elif args.action == 'download-repo':
        download_repo_from_hf(args.repo_id, args.repo_type, args.local_path)

