"""
Inference script for DeepVQE model

This script loads a trained checkpoint and performs inference on test audio files.
It handles STFT transformation, model inference, and inverse STFT with audio saving.

Usage:
    # Inference on a directory of test files
    python inference.py --checkpoint ./checkpoints/best.ckpt \
                        --test_dir ./test_audio \
                        --output_dir ./enhanced_audio

    # Inference on specific files
    python inference.py --checkpoint ./checkpoints/best.ckpt \
                        --mic_file test_mic.wav \
                        --farend_file test_farend.wav \
                        --output_file enhanced.wav

    # Download checkpoint from HuggingFace Hub
    python inference.py --hf_repo_id username/deep-enhancer-checkpoints \
                        --checkpoint_name best.ckpt \
                        --test_dir ./test_audio
"""
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
from deepvqe import DeepVQE_S, DeepVQE
from hf_hub_utils import HFCheckpointUploader


def load_checkpoint(checkpoint_path, model_class=DeepVQE_S, device='auto'):
    """
    Load a trained checkpoint and return the model

    Args:
        checkpoint_path: Path to checkpoint file (.ckpt)
        model_class: Model class to instantiate (DeepVQE_S or DeepVQE)
        device: Device to load model on ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        model: Loaded model in eval mode
        device: Device the model is on
    """
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model state dict (pure PyTorch checkpoint format)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Assume direct state dict
        state_dict = checkpoint

    # Handle torch.compile() wrapper - remove "_orig_mod." prefix
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("  Detected torch.compile() checkpoint, removing '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Create model and load weights
    model = model_class().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Print checkpoint info
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        elif 'best_val_loss' in checkpoint:
            print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model, device


def audio_to_stft(
    audio_path,
    sample_rate=24000,
    n_fft=512,
    hop_length=128,
    win_length=512,
    device='cpu'
):
    """
    Load audio file and convert to STFT

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        device: Device to put tensor on

    Returns:
        stft_complex: STFT tensor of shape (F, T, 2) [real, imag]
        original_length: Original audio length in samples (for iSTFT)
        original_sr: Original sample rate of the input audio
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    original_sr = sr  # Store original sample rate

    # Mono conversion
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    original_length = waveform.shape[1]
    waveform = waveform.squeeze(0)

    # STFT transform
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=None,
        return_complex=True
    )

    # Compute STFT
    stft = stft_transform(waveform)  # Complex tensor (F, T)

    # Convert to (F, T, 2) format [real, imag]
    stft_complex = torch.stack([stft.real, stft.imag], dim=-1)

    return stft_complex.to(device), original_length, original_sr


def stft_to_audio(
    stft_complex,
    output_path,
    sample_rate=24000,
    n_fft=512,
    hop_length=128,
    win_length=512,
    original_length=None,
    original_sr=None
):
    """
    Convert STFT back to audio and save

    Args:
        stft_complex: STFT tensor of shape (F, T, 2) [real, imag]
        output_path: Path to save audio file
        sample_rate: Sample rate (processing rate, typically 24kHz)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        original_length: Original audio length (for trimming)
        original_sr: Original sample rate to resample back to (if different from sample_rate)
    """
    # Convert (F, T, 2) back to complex tensor
    stft = torch.complex(stft_complex[..., 0], stft_complex[..., 1])

    # Inverse STFT
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )

    waveform = istft_transform(stft)

    # Trim to original length if provided
    if original_length is not None:
        waveform = waveform[:original_length]

    # Resample back to original sample rate if different
    if original_sr is not None and original_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, original_sr)
        waveform = resampler(waveform)

    # Save audio
    waveform = waveform.unsqueeze(0).cpu()  # Add channel dimension
    save_sr = original_sr if original_sr is not None else sample_rate
    torchaudio.save(output_path, waveform, save_sr)


@torch.no_grad()
def inference_single(
    model,
    mic_path,
    farend_path,
    output_path,
    device,
    sample_rate=24000,
    n_fft=512,
    hop_length=128,
    win_length=512
):
    """
    Run inference on a single pair of audio files

    Args:
        model: DeepVQE model
        mic_path: Path to noisy microphone audio
        farend_path: Path to far-end reference audio
        output_path: Path to save enhanced audio
        device: Device to run inference on
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
    """
    # Load and convert to STFT
    mic_stft, original_length, original_sr = audio_to_stft(
        mic_path, sample_rate, n_fft, hop_length, win_length, device
    )
    farend_stft, _, _ = audio_to_stft(
        farend_path, sample_rate, n_fft, hop_length, win_length, device
    )

    # Align time dimensions by padding to match the longer one
    # This handles cases where mic and farend have slightly different lengths
    max_time = max(mic_stft.shape[1], farend_stft.shape[1])

    if mic_stft.shape[1] != farend_stft.shape[1]:
        # Pad the shorter one with zeros
        if mic_stft.shape[1] < max_time:
            pad_size = max_time - mic_stft.shape[1]
            mic_stft = torch.nn.functional.pad(mic_stft, (0, 0, 0, pad_size))
        if farend_stft.shape[1] < max_time:
            pad_size = max_time - farend_stft.shape[1]
            farend_stft = torch.nn.functional.pad(farend_stft, (0, 0, 0, pad_size))

    # Add batch dimension: (F, T, 2) -> (1, F, T, 2)
    mic_stft = mic_stft.unsqueeze(0)
    farend_stft = farend_stft.unsqueeze(0)

    # Model inference
    enhanced_stft = model(mic_stft, farend_stft)

    # Remove batch dimension: (1, F, T, 2) -> (F, T, 2)
    enhanced_stft = enhanced_stft.squeeze(0)

    # Convert back to audio and save (resample back to original sample rate)
    stft_to_audio(
        enhanced_stft.cpu(),
        output_path,
        sample_rate,
        n_fft,
        hop_length,
        win_length,
        original_length,
        original_sr
    )


def inference_directory(
    model,
    test_dir,
    output_dir,
    device,
    sample_rate=24000,
    n_fft=512,
    hop_length=128,
    win_length=512
):
    """
    Run inference on all audio files in a directory

    Expected structure:
        test_dir/
            f00000_mic.wav
            f00000_farend.wav
            f00001_mic.wav
            f00001_farend.wav
            ...

    Args:
        model: DeepVQE model
        test_dir: Directory containing test audio files
        output_dir: Directory to save enhanced audio
        device: Device to run inference on
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
    """
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all mic files
    mic_files = sorted(test_dir.glob('*_mic.wav'))

    if len(mic_files) == 0:
        print(f"No audio files found in {test_dir}")
        print("Expected files like: f00000_mic.wav, f00000_farend.wav")
        return

    print(f"\nFound {len(mic_files)} test samples in {test_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Process all files
    failed = []
    for mic_file in tqdm(mic_files, desc="Processing audio files"):
        sample_id = mic_file.stem.replace('_mic', '')

        try:
            # Define paths
            mic_path = test_dir / f"{sample_id}_mic.wav"
            farend_path = test_dir / f"{sample_id}_farend.wav"
            output_path = output_dir / f"{sample_id}_enhanced.wav"

            # Check both files exist
            if not farend_path.exists():
                tqdm.write(f"Missing farend file for {sample_id}, skipping")
                failed.append(sample_id)
                continue

            # Run inference
            inference_single(
                model,
                mic_path,
                farend_path,
                output_path,
                device,
                sample_rate,
                n_fft,
                hop_length,
                win_length
            )

        except Exception as e:
            tqdm.write(f"Error processing {sample_id}: {e}")
            failed.append(sample_id)

    print(f"\n{'='*80}")
    print("Inference Complete!")
    print(f"{'='*80}")
    print(f"Processed: {len(mic_files) - len(failed)}/{len(mic_files)} samples")
    print(f"Enhanced audio saved to: {output_dir}")

    if failed:
        print(f"\nFailed samples ({len(failed)}): {failed[:10]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


def resolve_checkpoint_path(args):
    """
    Resolve checkpoint path from args (download from HF or find locally)

    Returns:
        checkpoint_path: Path to checkpoint file, or None if not found
    """
    # Option 1: Download from HuggingFace Hub
    if args.hf_repo_id and args.checkpoint_name:
        print(f"Downloading checkpoint from HuggingFace Hub...")
        print(f"  Repo: {args.hf_repo_id}")
        print(f"  Checkpoint: {args.checkpoint_name}\n")

        uploader = HFCheckpointUploader(
            repo_id=args.hf_repo_id,
            token=args.hf_token,
            auto_create_repo=False
        )

        checkpoint_path = uploader.download_checkpoint(
            checkpoint_name=args.checkpoint_name,
            local_dir=args.checkpoint_dir
        )

        if checkpoint_path:
            print()
        return checkpoint_path

    # Option 2: Use specified checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            return str(checkpoint_path)
        else:
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return None

    # Option 3: Search for best checkpoint in checkpoint_dir
    print(f"Searching for checkpoint in {args.checkpoint_dir}...")
    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Look for checkpoints with "best" in the name
    best_ckpts = sorted(checkpoint_dir.glob('best*.ckpt'))
    if best_ckpts:
        return str(best_ckpts[0])

    # Fallback: look for any .ckpt files
    all_ckpts = sorted(checkpoint_dir.glob('*.ckpt'))
    if all_ckpts:
        return str(all_ckpts[0])

    return None


def main():
    parser = argparse.ArgumentParser(description='DeepVQE Inference')

    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to search for checkpoints (if --checkpoint not specified)')
    parser.add_argument('--model', type=str, default='DeepVQE_S',
                        choices=['DeepVQE_S', 'DeepVQE'],
                        help='Model architecture (must match checkpoint)')

    # HuggingFace Hub arguments
    parser.add_argument('--hf_repo_id', type=str, default=None,
                        help='HF repo ID (e.g., username/deep-enhancer-checkpoints)')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Checkpoint filename in HF repo (requires --hf_repo_id)')

    # Input/Output arguments
    parser.add_argument('--test-dir', type=str, default=None,
                        help='Directory with test audio files (batch mode)')
    parser.add_argument('--output-dir', type=str, default='./enhanced_audio',
                        help='Output directory for enhanced audio')
    parser.add_argument('--mic-file', type=str, default=None,
                        help='Single mic audio file (single mode)')
    parser.add_argument('--farend-file', type=str, default=None,
                        help='Single farend audio file (single mode)')
    parser.add_argument('--output-file', type=str, default='enhanced.wav',
                        help='Output file for single mode')

    # STFT parameters (must match training)
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='Sample rate (24kHz)')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT size')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Hop length')
    parser.add_argument('--win_length', type=int, default=512,
                        help='Window length')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to run inference on')

    args = parser.parse_args()

    print(f"{'='*80}")
    print("DeepVQE Inference")
    print(f"{'='*80}\n")

    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint_path(args)
    if not checkpoint_path:
        print("Error: No checkpoint found")
        print("\nOptions:")
        print("  1. Specify: --checkpoint path/to/checkpoint.ckpt")
        print("  2. Download: --hf_repo_id username/repo --checkpoint_name file.ckpt")
        print("  3. Auto-find: Place checkpoint in ./checkpoints/")
        return

    print(f"Using checkpoint: {checkpoint_path}\n")

    # Load model
    model_class = DeepVQE_S if args.model == 'DeepVQE_S' else DeepVQE
    model, device = load_checkpoint(checkpoint_path, model_class, args.device)

    print(f"\nSTFT parameters:")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  n_fft: {args.n_fft}")
    print(f"  hop_length: {args.hop_length}")
    print(f"  win_length: {args.win_length}")

    # Run inference
    if args.test_dir:
        # Batch mode
        inference_directory(
            model, args.test_dir, args.output_dir, device,
            args.sample_rate, args.n_fft, args.hop_length, args.win_length
        )
    elif args.mic_file and args.farend_file:
        # Single file mode
        print(f"\nProcessing single file...")
        print(f"  Mic: {args.mic_file}")
        print(f"  Farend: {args.farend_file}")
        print(f"  Output: {args.output_file}")

        inference_single(
            model, args.mic_file, args.farend_file, args.output_file, device,
            args.sample_rate, args.n_fft, args.hop_length, args.win_length
        )

        print(f"\n{'='*80}")
        print(f"Enhanced audio saved to: {args.output_file}")
        print(f"{'='*80}")
    else:
        print("Error: Must specify either:")
        print("  --test_dir (batch mode)")
        print("  OR --mic_file and --farend_file (single mode)")


if __name__ == "__main__":
    main()
