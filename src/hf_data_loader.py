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
    """Download one or more files from a Hugging Face Hub repository.

    Args:
        filename: Single filename (str) or list of filenames to download
        hf_repo_id: Repository ID (e.g., 'username/repo-name')
        hf_repo_type: Type of repo ('dataset', 'model', or 'space')
        local_dir: Local directory to save the file(s) (default: current directory)

    Returns:
        str or list: Path(s) to the downloaded file(s)
    """
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    # Handle single file or multiple files
    if isinstance(filename, str):
        filenames = [filename]
        return_single = True
    else:
        filenames = filename
        return_single = False

    downloaded_paths = []
    for fn in filenames:
        print(f"Downloading: {fn}")
        file_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=fn,
            repo_type=hf_repo_type,
            token=token,
            local_dir=local_dir
        )
        print(f"  → {file_path}")
        downloaded_paths.append(file_path)

    return downloaded_paths[0] if return_single else downloaded_paths

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
    parser.add_argument('--remote-filepath', type=str, nargs='+',
                        help='Filename(s) to download from the repository (for download-file action). Can specify multiple files.')
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
        if not args.remote_filepath:
            parser.error('--remote-filepath is required for download-file')
        # Pass list if multiple files, single string if one file
        files = args.remote_filepath[0] if len(args.remote_filepath) == 1 else args.remote_filepath
        download_file_from_hf(files, args.repo_id, args.repo_type, args.local_path)
    elif args.action == 'download-repo':
        download_repo_from_hf(args.repo_id, args.repo_type, args.local_path)

