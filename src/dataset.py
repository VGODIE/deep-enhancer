import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from pathlib import Path


class DeepVQEDataset(Dataset):
    """
    Dataset for DeepVQE training with AEC + Speech Enhancement

    Expected data structure:
        data/
            f00000_mic.wav       - noisy microphone (mic + echo + noise)
            f00000_farend.wav    - far-end reference (system playback)
            f00000_target.wav    - clean target
            f00001_mic.wav
            ...
    """
    def __init__(
        self,
        data_dir='./data',
        n_fft=512,
        hop_length=128,
        win_length=512,
        sample_rate=24000,  # Downsample to 24kHz as per DeepVQE paper
        chunk_length_sec=10.0,  # Actual duration: 10 seconds
        cache_stft=False
    ):
        """
        Args:
            data_dir: Directory containing wav files
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            sample_rate: Expected sample rate
            chunk_length_sec: Length of audio chunks in seconds (None = full audio)
            cache_stft: Cache STFT in memory (uses more RAM but faster training)
        """
        self.data_dir = Path(data_dir)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.chunk_length_sec = chunk_length_sec
        self.cache_stft = cache_stft

        # STFT transform
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,  # Return complex spectrogram
            return_complex=True
        )

        # Find all sample IDs
        mic_files = sorted(self.data_dir.glob('*_mic.wav'))
        self.sample_ids = [f.stem.replace('_mic', '') for f in mic_files]

        print(f"Found {len(self.sample_ids)} samples in {data_dir}")
        print(f"STFT config: n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}")

        # Optional: cache for faster training
        self.cache = {} if cache_stft else None

    def __len__(self):
        return len(self.sample_ids)

    def _load_audio(self, file_path):
        """Load audio file and resample if necessary"""
        waveform, sr = torchaudio.load(file_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # (T,)

    def _audio_to_complex_stft(self, waveform):
        """
        Convert waveform to complex STFT in DeepVQE format

        Args:
            waveform: (T,) audio tensor

        Returns:
            stft_complex: (F, T, 2) tensor where [..., 0]=real, [..., 1]=imag
        """
        # Compute complex STFT: (F, T)
        stft = self.stft_transform(waveform)

        # Convert complex tensor to (F, T, 2) format
        stft_real = stft.real
        stft_imag = stft.imag
        stft_complex = torch.stack([stft_real, stft_imag], dim=-1)  # (F, T, 2)

        return stft_complex

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'noisy_mic': (F, T, 2) - noisy microphone STFT
                - 'farend': (F, T, 2) - far-end reference STFT
                - 'target': (F, T, 2) - clean target STFT
                - 'sample_id': str - sample identifier
        """
        sample_id = self.sample_ids[idx]

        # Check cache first
        if self.cache is not None and sample_id in self.cache:
            return self.cache[sample_id]

        # Load audio files
        mic_path = self.data_dir / f"{sample_id}_mic.wav"
        farend_path = self.data_dir / f"{sample_id}_farend.wav"
        target_path = self.data_dir / f"{sample_id}_target.wav"

        mic_audio = self._load_audio(mic_path)
        farend_audio = self._load_audio(farend_path)
        target_audio = self._load_audio(target_path)

        # Optional: chunk audio to fixed length
        if self.chunk_length_sec is not None:
            chunk_samples = int(self.chunk_length_sec * self.sample_rate)

            # Pad or truncate
            def pad_or_truncate(audio, length):
                if audio.shape[0] > length:
                    # Random crop during training
                    start = torch.randint(0, audio.shape[0] - length + 1, (1,)).item()
                    return audio[start:start + length]
                else:
                    # Pad if too short
                    return torch.nn.functional.pad(audio, (0, length - audio.shape[0]))

            mic_audio = pad_or_truncate(mic_audio, chunk_samples)
            farend_audio = pad_or_truncate(farend_audio, chunk_samples)
            target_audio = pad_or_truncate(target_audio, chunk_samples)

        # Convert to STFT
        noisy_mic_stft = self._audio_to_complex_stft(mic_audio)
        farend_stft = self._audio_to_complex_stft(farend_audio)
        target_stft = self._audio_to_complex_stft(target_audio)

        sample = {
            'noisy_mic': noisy_mic_stft,  # (F, T, 2)
            'farend': farend_stft,         # (F, T, 2)
            'target': target_stft,         # (F, T, 2)
            'sample_id': sample_id
        }

        # Cache if enabled
        if self.cache is not None:
            self.cache[sample_id] = sample

        return sample


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
    data_dir='./data',
    batch_size=4,
    train_split=0.9,
    num_workers=4,
    n_fft=512,
    hop_length=128,
    chunk_length_sec=10.0,  # Updated to match actual dataset (10 seconds)
    pin_memory=False,  # Set to False by default (can be overridden for CUDA)
    use_fast_loader=True,  # Use pre-computed STFT if available
    **kwargs
):
    """
    Create train and validation dataloaders

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of data loading workers
        n_fft: FFT size
        hop_length: STFT hop length
        chunk_length_sec: Audio chunk length in seconds
        pin_memory: Pin memory for faster GPU transfer (set False for MPS)
        use_fast_loader: Use FastDeepVQEDataset (pre-computed STFT) if available

    Returns:
        train_loader, val_loader
    """
    # Try to use fast loader first
    if use_fast_loader:
        # Try multiple possible locations for preprocessed data
        stft_dir_options = [
            Path('./data_stft'),
            Path(data_dir).parent / 'data_stft',
            Path(str(data_dir) + '_stft')
        ]

        stft_dir = None
        for option in stft_dir_options:
            if option.exists() and any(option.glob('*.pt')):
                stft_dir = option
                break

        if stft_dir:
            print(f"Using FastDeepVQEDataset (pre-computed STFT) from {stft_dir}")
            dataset = FastDeepVQEDataset(data_dir=stft_dir)
        else:
            print(f"\nWARNING: Pre-computed STFT not found!")
            print(f"Looked in: {', '.join(str(d) for d in stft_dir_options)}")
            print("Using regular dataset with on-the-fly STFT computation (SLOW)")
            print("\nTo speed up training 10-20x, run:")
            print("  python preprocess_dataset.py --data_dir ./data --output_dir ./data_stft\n")
            dataset = DeepVQEDataset(
                data_dir=data_dir,
                n_fft=n_fft,
                hop_length=hop_length,
                chunk_length_sec=chunk_length_sec,
                **kwargs
            )
    else:
        # Create full dataset with on-the-fly STFT
        dataset = DeepVQEDataset(
            data_dir=data_dir,
            n_fft=n_fft,
            hop_length=hop_length,
            chunk_length_sec=chunk_length_sec,
            **kwargs
        )

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


if __name__ == "__main__":
    # Test the dataset
    print("Testing DeepVQE Dataset...")

    dataset = DeepVQEDataset(
        data_dir='./data',
        chunk_length_sec=4.0
    )

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
        data_dir='./data',
        batch_size=4,
        num_workers=2
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  noisy_mic: {batch['noisy_mic'].shape}")
    print(f"  farend: {batch['farend'].shape}")
    print(f"  target: {batch['target'].shape}")
    print(f"  sample_ids: {len(batch['sample_ids'])} samples")
