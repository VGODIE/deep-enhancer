import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import webdataset as wds
import io
import os


class FastDeepVQEDataset(Dataset):
    """
    Fast dataset loader for pre-computed STFT files

    Expected data structure:
        data_stft/
            f00000.pt  - Contains {noisy_mic, farend, target, sample_id}
            f00001.pt
            ...

    This is 10-20x faster than computing STFT on-the-fly!
    Run preprocess_dataset.py first to generate these files.
    """
    def __init__(self, data_dir='./data_stft'):
        self.data_dir = Path(data_dir)
        self.sample_files = sorted(self.data_dir.glob('*.pt'))

        if len(self.sample_files) == 0:
            raise ValueError(
                f"No .pt files found in {data_dir}!\n"
                f"Run: python preprocess_dataset.py --data_dir ./data --output_dir {data_dir}"
            )

        print(f"FastDeepVQEDataset: Found {len(self.sample_files)} pre-computed samples in {data_dir}")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        """Load pre-computed STFT from disk (very fast!)"""
        sample = torch.load(self.sample_files[idx])
        return sample


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences

    Args:
        batch: List of dicts from __getitem__

    Returns:
        Batched dict with tensors of shape (B, F, T, 2)
    """
    # Stack all samples
    noisy_mic = torch.stack([item['noisy_mic'] for item in batch])  # (B, F, T, 2)
    farend = torch.stack([item['farend'] for item in batch])
    target = torch.stack([item['target'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]

    return {
        'noisy_mic': noisy_mic,
        'farend': farend,
        'target': target,
        'sample_ids': sample_ids
    }


def create_dataloaders(
    data_dir='./data_stft',
    batch_size=4,
    train_split=0.9,
    num_workers=4,
    pin_memory=False,  # Set to False by default (can be overridden for CUDA)
):
    """
    Create train and validation dataloaders

    Args:
        data_dir: Path to pre-computed STFT data directory (default: './data_stft')
        batch_size: Batch size
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer (set False for MPS)

    Returns:
        train_loader, val_loader
    """
    print(f"Using FastDeepVQEDataset (pre-computed STFT) from {data_dir}")
    dataset = FastDeepVQEDataset(data_dir=data_dir)

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Create dataloaders
    # Note: pin_memory should be False for MPS (Mac M1/M2), True for CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"Train samples: {train_size}, Val samples: {val_size}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader


def decode_sample(sample):
    """Decode a sample from webdataset tar archive"""
    # Skip samples that don't have all required .pt files
    if 'noisy_mic.pt' not in sample or 'farend.pt' not in sample or 'target.pt' not in sample:
        return None

    sample_id = sample['__key__']
    noisy_mic = torch.load(io.BytesIO(sample['noisy_mic.pt']))
    farend = torch.load(io.BytesIO(sample['farend.pt']))
    target = torch.load(io.BytesIO(sample['target.pt']))

    return {
        'noisy_mic': noisy_mic,
        'farend': farend,
        'target': target,
        'sample_id': sample_id
    }


def create_dataloaders_wds(
    hf_repo_id=None,
    train_tar_pattern='train-*.tar.gz',
    val_tar_pattern='val-*.tar.gz',
    batch_size=4,
    num_workers=4,
    pin_memory=False,
    hf_token=None,
    cache_dir=None,
):
    """
    Create dataloaders using webdataset from local or HuggingFace Hub tar.gz archives

    Args:
        hf_repo_id: HuggingFace repo ID (e.g., 'username/dataset'). If None, uses local paths
        train_tar_pattern: String or list of patterns/files
            - For local: './data/train-*.tar.gz' or ['./data/train-000.tar.gz']
            - For HF: 'train-*.tar.gz' or ['train-000.tar.gz', 'train-001.tar.gz']
        val_tar_pattern: String or list of patterns/files
    """
    from dotenv import load_dotenv
    load_dotenv()

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    # Determine if using local files or HuggingFace Hub
    if hf_repo_id is None:
        # Local mode - expand glob patterns
        import glob
        print("Creating WebDataset from local tar files")

        if isinstance(train_tar_pattern, str):
            train_urls = sorted(glob.glob(train_tar_pattern))
            if not train_urls:
                raise FileNotFoundError(f"No files found matching pattern: {train_tar_pattern}")
        else:
            train_urls = []
            for pattern in train_tar_pattern:
                matched = sorted(glob.glob(pattern))
                if not matched:
                    print(f"Warning: No files found for pattern: {pattern}")
                train_urls.extend(matched)
            if not train_urls:
                raise FileNotFoundError(f"No files found matching any patterns: {train_tar_pattern}")

        if isinstance(val_tar_pattern, str):
            val_urls = sorted(glob.glob(val_tar_pattern))
            if not val_urls:
                raise FileNotFoundError(f"No files found matching pattern: {val_tar_pattern}")
        else:
            val_urls = []
            for pattern in val_tar_pattern:
                matched = sorted(glob.glob(pattern))
                if not matched:
                    print(f"Warning: No files found for pattern: {pattern}")
                val_urls.extend(matched)
            if not val_urls:
                raise FileNotFoundError(f"No files found matching any patterns: {val_tar_pattern}")

        print(f"Found {len(train_urls)} training tar files:")
        for f in train_urls[:5]:
            print(f"  - {f}")
        if len(train_urls) > 5:
            print(f"  ... and {len(train_urls) - 5} more")

        print(f"Found {len(val_urls)} validation tar files:")
        for f in val_urls[:5]:
            print(f"  - {f}")
        if len(val_urls) > 5:
            print(f"  ... and {len(val_urls) - 5} more")
    else:
        # HuggingFace Hub mode
        base_url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main"

        if isinstance(train_tar_pattern, str):
            train_urls = f"{base_url}/{train_tar_pattern}"
        else:
            train_urls = [f"{base_url}/{p}" for p in train_tar_pattern]

        if isinstance(val_tar_pattern, str):
            val_urls = f"{base_url}/{val_tar_pattern}"
        else:
            val_urls = [f"{base_url}/{p}" for p in val_tar_pattern]

        print(f"Creating WebDataset from HuggingFace Hub: {hf_repo_id}")
        print(f"Train: {train_tar_pattern}, Val: {val_tar_pattern}")

    # Don't batch in the dataset - let the DataLoader handle it with collate_fn
    train_dataset = (
        wds.WebDataset(train_urls, cache_dir=cache_dir, shardshuffle=True, empty_check=False)
        .shuffle(1000)
        .decode()
        .map(decode_sample)
        .select(lambda x: x is not None)
    )

    val_dataset = (
        wds.WebDataset(val_urls, cache_dir=cache_dir, shardshuffle=False, empty_check=False)
        .decode()
        .map(decode_sample)
        .select(lambda x: x is not None)
    )

    # Use WebLoader with batch_size and collate_fn (same as regular DataLoader)
    train_loader = wds.WebLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = wds.WebLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    print("WebDataset dataloaders created successfully!")
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the fast dataset
    print("Testing FastDeepVQE Dataset...")

    dataset = FastDeepVQEDataset(data_dir='./data_stft')

    # Load one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  noisy_mic: {sample['noisy_mic'].shape}")
    print(f"  farend: {sample['farend'].shape}")
    print(f"  target: {sample['target'].shape}")
    print(f"  sample_id: {sample['sample_id']}")

    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = create_dataloaders(
        data_dir='./data_stft',
        batch_size=4,
        num_workers=2
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  noisy_mic: {batch['noisy_mic'].shape}")
    print(f"  farend: {batch['farend'].shape}")
    print(f"  target: {batch['target'].shape}")
    print(f"  sample_ids: {len(batch['sample_ids'])} samples")
