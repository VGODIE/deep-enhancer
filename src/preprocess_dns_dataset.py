"""
Pre-process DNS Challenge dataset for nearend single talk scenarios
Directly outputs STFT-encoded .pt files (no intermediate WAV files)

This script mixes clean speech with noise to create nearend singletalk:
1. Create noisy-mic: clean speech mixed with noise
2. Create triplet in memory: noisy-mic, silence (farend), clean (target)
3. Transform all three to STFT
4. Save as single .pt file with sample_id

Audio handling:
- Clips longer than 10 seconds are trimmed to 10 seconds (first chunk)
- Clips shorter than 10 seconds are padded IF they are at least 8 seconds
- Clips shorter than 8 seconds are skipped

Mixing strategy (from DNS Challenge):
- Random SNR sampling between snr_lower and snr_upper (default: -5 to 20 dB)
- Activity detection to filter out silent clips
- Segmental SNR mixing for realistic noise levels

Usage:
    python preprocess_dns_dataset.py \
        --clean-dir ./dns_clean \
        --noise-dir ./dns_noise \
        --output-dir ./data_stft_dns \
        --num-samples 10000

Output structure:
    data_stft_dns/
        dns_nearend_000000_snr10.pt
        dns_nearend_000001_snr15.pt
        ...

Dataset structure expected:
    dns_clean/
        clean_fileid_0.wav
        clean_fileid_1.wav
        ...
    dns_noise/
        noise_fileid_0.wav
        noise_fileid_1.wav
        ...
"""
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import random


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def activitydetector(audio, fs=24000, energy_thresh=0.13):
    """
    Activity detector based on DNS Challenge code

    Args:
        audio: Audio waveform (torch tensor or numpy array)
        fs: Sample rate
        energy_thresh: Energy threshold for activity detection

    Returns:
        percentage of active frames
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    # Ensure 1D
    if len(audio.shape) > 1:
        audio = audio.squeeze()

    # Frame parameters
    frame_len = int(0.025 * fs)  # 25ms frames
    hop_len = int(0.010 * fs)    # 10ms hop

    # Compute energy
    num_frames = (len(audio) - frame_len) // hop_len + 1

    if num_frames <= 0:
        return 0.0

    active_frames = 0
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frame = audio[start:end]

        # Compute frame energy
        energy = np.sum(frame ** 2) / len(frame)

        if energy > energy_thresh:
            active_frames += 1

    return active_frames / num_frames


def segmental_snr_mixer(clean, noise, snr, target_level=-25, eps=1e-10):
    """
    Mix clean speech with noise at specified SNR
    Based on DNS Challenge mixing code

    Args:
        clean: Clean speech waveform (torch tensor)
        noise: Noise waveform (torch tensor)
        snr: Target SNR in dB
        target_level: Target level in dB
        eps: Small value for numerical stability

    Returns:
        clean_scaled, noise_scaled, noisy, target_level_used
    """
    # Convert to numpy for processing
    clean_np = clean.cpu().numpy() if isinstance(clean, torch.Tensor) else clean
    noise_np = noise.cpu().numpy() if isinstance(noise, torch.Tensor) else noise

    # Ensure same length
    if len(noise_np) < len(clean_np):
        # Repeat noise if too short
        num_repeats = int(np.ceil(len(clean_np) / len(noise_np)))
        noise_np = np.tile(noise_np, num_repeats)[:len(clean_np)]
    else:
        noise_np = noise_np[:len(clean_np)]

    # Normalize to target level
    clean_rms = np.sqrt(np.mean(clean_np ** 2)) + eps
    clean_scaled = clean_np * (10 ** (target_level / 20) / clean_rms)

    # Compute noise scaling for target SNR
    noise_rms = np.sqrt(np.mean(noise_np ** 2)) + eps
    noise_level = target_level - snr
    noise_scaled = noise_np * (10 ** (noise_level / 20) / noise_rms)

    # Mix
    noisy = clean_scaled + noise_scaled

    # Avoid clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        scale = 0.95 / max_val
        clean_scaled *= scale
        noise_scaled *= scale
        noisy *= scale

    # Convert back to torch tensors
    clean_scaled = torch.from_numpy(clean_scaled).float()
    noise_scaled = torch.from_numpy(noise_scaled).float()
    noisy = torch.from_numpy(noisy).float()

    return clean_scaled, noise_scaled, noisy, target_level


def load_and_clip_audio(file_path, sample_rate=24000, target_length_sec=10.0, device='cpu'):
    """
    Load audio file, clip to target length (exactly 10 seconds)

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        target_length_sec: Target length in seconds (default 10)
        device: Device to use

    Returns:
        waveform (torch tensor) of exactly target_length_sec, or None if failed
    """
    try:
        waveform, sr = torchaudio.load(file_path)

        # Move to device
        waveform = waveform.to(device)

        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate).to(device)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        target_length = int(target_length_sec * sample_rate)

        # Check if file is long enough
        if len(waveform) < target_length:
            return None  # Skip files shorter than target

        # Clip to exactly target length (take first 10 seconds)
        waveform = waveform[:target_length]

        return waveform

    except Exception:
        return None


def get_random_audio_clip(source_files, sample_rate=24000, target_length_sec=10.0, device='cpu',
                          activity_threshold=0.0, max_tries=100):
    """
    Pick a random audio file and clip it to target length
    Optionally check for activity to skip silent clips

    Args:
        source_files: List of source audio files
        sample_rate: Target sample rate
        target_length_sec: Target length in seconds
        device: Device to use
        activity_threshold: Minimum activity level (0.0 = no check)
        max_tries: Maximum number of files to try

    Returns:
        audio clip of exactly target_length_sec, or None if failed after max_tries
    """
    for _ in range(max_tries):
        # Pick random file
        file_path = random.choice(source_files)

        # Try to load and clip
        audio = load_and_clip_audio(file_path, sample_rate, target_length_sec, device)

        if audio is None:
            continue

        # Check activity if threshold is set
        if activity_threshold > 0.0:
            activity = activitydetector(audio, sample_rate)
            if activity < activity_threshold:
                continue  # Skip silent clips

        return audio

    return None


def convert_to_stft(waveform, stft_transform):
    """
    Convert waveform to STFT format

    Args:
        waveform: Waveform tensor (1D)
        stft_transform: STFT transform object

    Returns:
        stft_complex: STFT tensor of shape (F, T, 2) [real, imag]
    """
    stft = stft_transform(waveform)  # Complex tensor (F, T)

    # Convert to (F, T, 2) format [real, imag]
    stft_complex = torch.stack([stft.real, stft.imag], dim=-1)

    return stft_complex.cpu()


def process_dns_sample(
    clean_files,
    noise_files,
    output_dir,
    stft_transform,
    sample_rate,
    chunk_length_sec,
    device='cpu',
    snr_lower=-5,
    snr_upper=20,
    target_level=-25,
    clean_activity_threshold=0.5,
    noise_activity_threshold=0.0,
    sample_counter=0
):
    """
    SIMPLIFIED: Process a single DNS sample
    1. Pick random clean audio (10 seconds) with activity check
    2. Pick random noise audio (10 seconds) with activity check
    3. Mix them with random SNR
    4. Transform to STFT
    5. Save as .pt file

    Args:
        clean_files: List of clean audio files
        noise_files: List of noise audio files
        output_dir: Output directory
        stft_transform: STFT transform
        sample_rate: Sample rate
        chunk_length_sec: Chunk length in seconds (10.0)
        device: Device to use
        snr_lower: Lower SNR bound (dB)
        snr_upper: Upper SNR bound (dB)
        target_level: Target level (dB)
        clean_activity_threshold: Min activity for clean (0.5 = 50%)
        noise_activity_threshold: Min activity for noise (0.0 = any)
        sample_counter: Sample number for unique ID

    Returns:
        True if successful, False otherwise
    """
    output_dir = Path(output_dir)

    # Step 1: Get random clean audio clip (exactly 10 seconds) with activity check
    clean = get_random_audio_clip(clean_files, sample_rate, chunk_length_sec, device,
                                   activity_threshold=clean_activity_threshold)
    if clean is None:
        return False

    # Step 2: Get random noise audio clip (exactly 10 seconds)
    noise = get_random_audio_clip(noise_files, sample_rate, chunk_length_sec, device,
                                   activity_threshold=noise_activity_threshold)
    if noise is None:
        return False

    # Step 3: Mix with random SNR
    snr = random.randint(snr_lower, snr_upper)
    clean_scaled, noise_scaled, noisy, _ = segmental_snr_mixer(
        clean, noise, snr, target_level
    )

    # Step 4: Create triplet (noisy-mic, silence, clean-target)
    noisy_mic = noisy.to(device)
    farend = torch.zeros_like(clean_scaled).to(device)
    target = clean_scaled.to(device)

    # Step 5: Transform to STFT
    try:
        noisy_mic_stft = convert_to_stft(noisy_mic, stft_transform)
        farend_stft = convert_to_stft(farend, stft_transform)
        target_stft = convert_to_stft(target, stft_transform)
    except Exception:
        return False

    # Step 6: Save
    output_id = f"dns_nearend_{sample_counter:06d}_snr{snr:+03d}"
    output_file = output_dir / f"{output_id}.pt"

    try:
        torch.save({
            'noisy_mic': noisy_mic_stft,
            'farend': farend_stft,
            'target': target_stft,
            'sample_id': output_id
        }, output_file)

        return True

    except Exception:
        if output_file.exists():
            output_file.unlink()
        return False


def preprocess_dns_nearend_singletalk(
    clean_dir='./dns_clean',
    noise_dir='./dns_noise',
    output_dir='./data_stft_dns',
    num_samples=10000,
    n_fft=512,
    hop_length=128,
    win_length=512,
    sample_rate=24000,
    chunk_length_sec=10.0,
    snr_lower=-5,
    snr_upper=20,
    target_level=-25,
    clean_activity_threshold=0.5,
    noise_activity_threshold=0.0,
    use_compile=True,
    seed=42
):
    """
    SIMPLIFIED: Pre-process DNS Challenge dataset for nearend single talk
    Directly outputs STFT-encoded .pt files (no intermediate WAV files)

    SIMPLIFIED LOGIC:
    1. Pick random clean audio file >= 10 seconds, clip to exactly 10 seconds
    2. Check activity level (skip if too silent)
    3. Pick random noise audio file >= 10 seconds, clip to exactly 10 seconds
    4. Mix them with random SNR
    5. Save STFT triplet to .pt file

    Args:
        clean_dir: Directory containing clean speech files
        noise_dir: Directory containing noise files
        output_dir: Output directory for STFT .pt files
        num_samples: Number of samples to generate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        sample_rate: Target sample rate
        chunk_length_sec: Chunk length in seconds (must be 10.0)
        snr_lower: Lower SNR bound (dB)
        snr_upper: Upper SNR bound (dB)
        target_level: Target level (dB)
        clean_activity_threshold: Min activity for clean (0.5 = skip silent clips)
        noise_activity_threshold: Min activity for noise (0.0 = accept all)
        use_compile: Use torch.compile for faster STFT
        seed: Random seed for reproducibility
    """
    set_seed(seed)

    clean_dir = Path(clean_dir)
    noise_dir = Path(noise_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Silicon (MPS)'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    print(f"{'='*80}")
    print(f"DNS Challenge Dataset Preprocessing (SIMPLIFIED)")
    print(f"Mode: Nearend Single Talk (Clean + Noise, Direct STFT output)")
    print(f"{'='*80}")
    print(f"Clean directory: {clean_dir}")
    print(f"Noise directory: {noise_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device_name}")
    print(f"STFT config: n_fft={n_fft}, hop={hop_length}, win={win_length}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk length: {chunk_length_sec} seconds (exactly)")
    print(f"  - Files >= 10s: clip to exactly 10 seconds")
    print(f"  - Files < 10s: skipped")
    print(f"Activity thresholds:")
    print(f"  - Clean: {clean_activity_threshold} (skip silent clips)")
    print(f"  - Noise: {noise_activity_threshold} (accept all)")
    print(f"SNR range: [{snr_lower}, {snr_upper}] dB")
    print(f"Target level: {target_level} dB")
    print(f"Samples to generate: {num_samples}")
    print(f"Random seed: {seed}")

    # STFT transform
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=None,
        return_complex=True
    ).to(device)

    # Apply torch.compile
    if use_compile:
        try:
            stft_transform = torch.compile(stft_transform, mode='reduce-overhead')
            print(f"torch.compile: Enabled (faster processing!)")
        except Exception:
            print(f"torch.compile: Not available (requires PyTorch 2.0+), using standard mode")
            use_compile = False
    else:
        print(f"torch.compile: Disabled")

    # Collect clean and noise files
    print("\nCollecting audio files...")
    clean_files = list(clean_dir.rglob('*.wav'))
    noise_files = list(noise_dir.rglob('*.wav'))

    print(f"Found {len(clean_files)} clean files")
    print(f"Found {len(noise_files)} noise files")

    if len(clean_files) == 0:
        raise ValueError(f"No clean files found in {clean_dir}")
    if len(noise_files) == 0:
        raise ValueError(f"No noise files found in {noise_dir}")

    # Shuffle
    random.shuffle(clean_files)
    random.shuffle(noise_files)

    # Convert to strings
    clean_files = [str(f) for f in clean_files]
    noise_files = [str(f) for f in noise_files]

    # Process samples
    print(f"\n{'='*80}")
    print("Generating samples...")
    print(f"Pipeline: Clean + Noise -> Noisy-Mic -> STFT triplet -> .pt file")
    print(f"{'='*80}\n")

    successful = 0
    failed = 0

    for i in tqdm(range(num_samples), desc="Generating DNS samples"):
        success = process_dns_sample(
            clean_files=clean_files,
            noise_files=noise_files,
            output_dir=output_dir,
            stft_transform=stft_transform,
            sample_rate=sample_rate,
            chunk_length_sec=chunk_length_sec,
            device=device,
            snr_lower=snr_lower,
            snr_upper=snr_upper,
            target_level=target_level,
            clean_activity_threshold=clean_activity_threshold,
            noise_activity_threshold=noise_activity_threshold,
            sample_counter=successful
        )

        if success:
            successful += 1
        else:
            failed += 1
            if failed <= 5:  # Show first 5 failures
                tqdm.write(f"Failed to generate sample {i} (not enough suitable audio files)")

    print(f"\n{'='*80}")
    print(f"Preprocessing Complete!")
    print(f"{'='*80}")
    print(f"\nStatistics:")
    print(f"  Successful samples: {successful}")
    print(f"  Failed samples: {failed}")
    print(f"  Success rate: {successful/num_samples*100:.1f}%")

    print(f"\nOutput directory: {output_dir}")

    # Verify output
    pt_files = list(output_dir.glob('*.pt'))
    print(f"\nVerification:")
    print(f"  .pt files: {len(pt_files)}")

    # Check one sample
    if pt_files:
        sample = torch.load(pt_files[0], weights_only=False)
        print(f"\nSample shape check (first file):")
        print(f"  noisy_mic: {sample['noisy_mic'].shape}")
        print(f"  farend: {sample['farend'].shape}")
        print(f"  target: {sample['target'].shape}")
        print(f"  sample_id: {sample['sample_id']}")

    # Print storage info
    try:
        processed_size = sum(f.stat().st_size for f in output_dir.glob('*.pt')) / 1e9
        print(f"\nStorage:")
        print(f"  Processed STFT files: {processed_size:.2f} GB")
        print(f"  Average per sample: {processed_size/len(pt_files)*1000:.2f} MB")
    except:
        pass

    print(f"\nNext steps:")
    print(f"  Train model:")
    print(f"     python train.py --use_fast_loader --data_dir {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess DNS Challenge dataset (nearend singletalk, direct STFT output)'
    )
    parser.add_argument('--clean-dir', type=str, required=True,
                        help='Directory containing clean speech files')
    parser.add_argument('--noise-dir', type=str, required=True,
                        help='Directory containing noise files')
    parser.add_argument('--output-dir', type=str, default='./data_stft_dns',
                        help='Output directory for STFT .pt files')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--n-fft', type=int, default=512,
                        help='FFT size')
    parser.add_argument('--hop-length', type=int, default=128,
                        help='Hop length')
    parser.add_argument('--win-length', type=int, default=512,
                        help='Window length')
    parser.add_argument('--sample-rate', type=int, default=24000,
                        help='Target sample rate')
    parser.add_argument('--chunk-sec', type=float, default=10.0,
                        help='Chunk length in seconds')
    parser.add_argument('--snr-lower', type=int, default=-5,
                        help='Lower SNR bound (dB)')
    parser.add_argument('--snr-upper', type=int, default=20,
                        help='Upper SNR bound (dB)')
    parser.add_argument('--target-level', type=int, default=-25,
                        help='Target level (dB)')
    parser.add_argument('--clean-activity', type=float, default=0.5,
                        help='Clean speech activity threshold (0.0-1.0, default 0.5)')
    parser.add_argument('--noise-activity', type=float, default=0.0,
                        help='Noise activity threshold (0.0-1.0, default 0.0)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile optimization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    preprocess_dns_nearend_singletalk(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        sample_rate=args.sample_rate,
        chunk_length_sec=args.chunk_sec,
        snr_lower=args.snr_lower,
        snr_upper=args.snr_upper,
        target_level=args.target_level,
        clean_activity_threshold=args.clean_activity,
        noise_activity_threshold=args.noise_activity,
        use_compile=not args.no_compile,
        seed=args.seed
    )