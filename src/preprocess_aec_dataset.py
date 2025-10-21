"""
Pre-process Microsoft AEC Challenge 2023 dataset for farend single talk scenarios
Directly outputs STFT-encoded .pt files (no intermediate WAV files)

This script converts the AEC Challenge farend singletalk dataset directly to STFT format:
- noisy_mic: STFT of mic audio (contains echo from farend)
- farend: STFT of loopback audio (reference)
- target: STFT of silence (ideal AEC removes all far-end speech)

Audio handling:
- Clips longer than 10 seconds are trimmed to 10 seconds (center chunk)
- Clips shorter than 10 seconds are padded IF they are at least 8 seconds
- Clips shorter than 8 seconds are skipped

Usage:
    python preprocess_aec_dataset.py \
        --aec_data_dir ./aec_challenge_2023 \
        --output_dir ./data_stft

Dataset structure expected:
    aec_challenge_2023/
        0a0aTELYCki1fyo5VvfMyQ_farend-singletalk_lpb.wav
        0a0aTELYCki1fyo5VvfMyQ_farend-singletalk_mic.wav
        zYAIKwD4BEysnGrzZuRXKQ_farend-singletalk-with-movement_lpb.wav
        zYAIKwD4BEysnGrzZuRXKQ_farend-singletalk-with-movement_mic.wav
        ...
"""
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict


def group_aec_files(aec_data_dir):
    """
    Group AEC Challenge files by sample ID and scenario type

    Returns:
        dict: {sample_id: {scenario_type: {lpb: path, mic: path}}}
    """
    aec_data_dir = Path(aec_data_dir)
    samples = defaultdict(lambda: defaultdict(dict))

    # Find all wav files
    wav_files = list(aec_data_dir.glob('*.wav'))
    print(f"Found {len(wav_files)} total WAV files")

    for wav_file in wav_files:
        filename = wav_file.stem

        # Only process farend singletalk files (with hyphen)
        if 'farend-singletalk' not in filename:
            continue

        # Parse filename: {sample_id}_{scenario}_{channel}.wav
        # Example: 0a0aTELYCki1fyo5VvfMyQ_farend-singletalk_lpb.wav
        parts = filename.split('_')

        if len(parts) < 3:
            continue

        sample_id = parts[0]
        channel = parts[-1]  # lpb or mic

        # Determine scenario type (with hyphen)
        if 'farend-singletalk-with-movement' in filename:
            scenario = 'farend-singletalk-with-movement'
        elif 'farend-singletalk' in filename:
            scenario = 'farend-singletalk'
        else:
            continue

        samples[sample_id][scenario][channel] = wav_file

    return samples


def load_and_convert_to_stft(
    file_path,
    stft_transform,
    sample_rate=24000,
    chunk_length_sec=10.0,
    device='cpu'
):
    """
    Load audio and convert to STFT format (same logic as preprocess_dataset.py)
    Now with GPU acceleration support

    Args:
        file_path: Path to audio file
        stft_transform: STFT transform object
        sample_rate: Target sample rate
        chunk_length_sec: Target chunk length in seconds (10.0)
        device: Device to use for computation ('cuda', 'cpu')

    Returns:
        stft_complex: STFT tensor of shape (F, T, 2) [real, imag], or None if too short
    """
    # Load audio
    waveform, sr = torchaudio.load(file_path)

    # Move to device for GPU acceleration
    waveform = waveform.to(device)

    # Mono conversion
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate).to(device)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    # Chunk to fixed length (IMPORTANT: ensures all samples same length!)
    chunk_samples = int(chunk_length_sec * sample_rate)
    min_required_samples = int(chunk_samples * 0.8)  # Require at least 80% of target length (8 sec)

    if waveform.shape[0] < min_required_samples:
        # Too short - skip this sample
        return None
    elif waveform.shape[0] > chunk_samples:
        # Take first chunk (first 10 seconds)
        waveform = waveform[:chunk_samples]
    elif waveform.shape[0] < chunk_samples:
        # Pad if slightly too short (but >= 8 seconds)
        waveform = torch.nn.functional.pad(waveform, (0, chunk_samples - waveform.shape[0]))

    # Compute STFT (on GPU if available)
    stft = stft_transform(waveform)  # Complex tensor (F, T)

    # Convert to (F, T, 2) format [real, imag]
    stft_complex = torch.stack([stft.real, stft.imag], dim=-1)

    # Move back to CPU for saving
    return stft_complex.cpu()


def create_silence_stft(reference_stft):
    """
    Create a silent STFT matching the shape of reference

    Args:
        reference_stft: Reference STFT tensor of shape (F, T, 2)

    Returns:
        silence_stft: Zero STFT tensor with same shape
    """
    return torch.zeros_like(reference_stft)


def process_farend_singletalk(
    sample_id,
    files,
    output_dir,
    stft_transform,
    sample_rate,
    chunk_length_sec,
    device='cpu',
    include_movement=True
):
    """
    Process farend singletalk samples and save as STFT .pt files

    Args:
        sample_id: Sample ID
        files: Dict of scenario files
        output_dir: Output directory
        stft_transform: STFT transform object
        sample_rate: Sample rate
        chunk_length_sec: Chunk length in seconds
        device: Device to use for computation
        include_movement: Include samples with movement

    Returns:
        list of created sample IDs
    """
    output_dir = Path(output_dir)
    created_samples = []

    # Scenarios to process (with hyphens)
    scenarios = ['farend-singletalk']
    if include_movement:
        scenarios.append('farend-singletalk-with-movement')

    for scenario in scenarios:
        if scenario in files:
            scenario_files = files[scenario]

            # Need both lpb and mic
            if 'lpb' not in scenario_files or 'mic' not in scenario_files:
                continue

            lpb_path = scenario_files['lpb']
            mic_path = scenario_files['mic']

            # Create output filename
            scenario_suffix = scenario.replace('farend-singletalk', 'farend')
            output_id = f"{sample_id}_{scenario_suffix}"

            # Convert to STFT (with GPU acceleration)
            mic_stft = load_and_convert_to_stft(
                mic_path, stft_transform, sample_rate, chunk_length_sec, device
            )
            farend_stft = load_and_convert_to_stft(
                lpb_path, stft_transform, sample_rate, chunk_length_sec, device
            )

            # Check if any are too short (None returned)
            if mic_stft is None or farend_stft is None:
                continue

            # Create silence for target (ideal AEC output)
            target_stft = create_silence_stft(mic_stft)

            # Save as single .pt file (same format as preprocess_dataset.py)
            # Wrap in try-except to skip corrupted files
            output_file = output_dir / f"{output_id}.pt"
            try:
                torch.save({
                    'noisy_mic': mic_stft,
                    'farend': farend_stft,
                    'target': target_stft,
                    'sample_id': output_id
                }, output_file)

                # Verify file can be loaded back
                _ = torch.load(output_file, weights_only=False)

                created_samples.append(output_id)
            except Exception as save_error:
                # Failed to save or verify - skip this sample
                if output_file.exists():
                    output_file.unlink()  # Clean up partial file
                continue

    return created_samples


def preprocess_aec_farend_singletalk(
    aec_data_dir='./aec_challenge_2023',
    output_dir='./data_stft',
    n_fft=512,
    hop_length=128,
    win_length=512,
    sample_rate=24000,
    chunk_length_sec=10.0,
    include_movement=True,
    use_compile=True
):
    """
    Pre-process AEC Challenge 2023 dataset (farend single talk only)
    Directly outputs STFT-encoded .pt files
    Now with GPU acceleration and torch.compile support!

    Args:
        aec_data_dir: Directory containing AEC Challenge WAV files
        output_dir: Output directory for STFT .pt files
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        sample_rate: Target sample rate (24kHz as per DeepVQE paper)
        chunk_length_sec: Chunk length in seconds (10.0 default)
        include_movement: Include samples with movement (default: True)
        use_compile: Use torch.compile for faster STFT (default: True, requires PyTorch 2.0+)
    """
    aec_data_dir = Path(aec_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Auto-detect device (CUDA, MPS, or CPU)
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
    print(f"Microsoft AEC Challenge 2023 Dataset Preprocessing")
    print(f"Mode: Farend Single Talk Only (Direct STFT output)")
    print(f"{'='*80}")
    print(f"Input directory: {aec_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device_name}")
    print(f"STFT config: n_fft={n_fft}, hop={hop_length}, win={win_length}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk length: {chunk_length_sec} seconds")
    print(f"  - Clips > 10s: trimmed to 10s (first 10 seconds)")
    print(f"  - Clips 8-10s: padded to 10s")
    print(f"  - Clips < 8s: skipped")
    print(f"Include movement: {include_movement}")

    # STFT transform
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=None,
        return_complex=True
    ).to(device)

    # Apply torch.compile for faster execution (PyTorch 2.0+)
    if use_compile:
        try:
            stft_transform = torch.compile(stft_transform, mode='reduce-overhead')
            print(f"torch.compile: Enabled (faster processing!)")
        except Exception as e:
            print(f"torch.compile: Not available (requires PyTorch 2.0+), using standard mode")
            use_compile = False
    else:
        print(f"torch.compile: Disabled")

    # Group files by sample ID
    print("\nGrouping files by sample ID...")
    samples = group_aec_files(aec_data_dir)
    print(f"Found {len(samples)} unique sample IDs with farend singletalk")

    # Process all samples
    print(f"\n{'='*80}")
    print("Processing samples (WAV -> STFT)...")
    print(f"{'='*80}\n")

    stats = {
        'farend_singletalk': 0,
        'farend_singletalk_with_movement': 0,
        'failed': 0,
        'too_short': 0
    }

    all_created = []

    for sample_id, scenarios in tqdm(samples.items(), desc="Converting WAV to STFT"):
        try:
            # Process farend singletalk (with device for GPU acceleration)
            farend_samples = process_farend_singletalk(
                sample_id, scenarios, output_dir,
                stft_transform, sample_rate, chunk_length_sec,
                device, include_movement
            )

            if len(farend_samples) == 0:
                stats['too_short'] += 1

            # Count by type
            for s in farend_samples:
                if 'movement' in s:
                    stats['farend_singletalk_with_movement'] += 1
                else:
                    stats['farend_singletalk'] += 1

            all_created.extend(farend_samples)

        except Exception as e:
            tqdm.write(f"Error processing {sample_id}: {str(e)[:100]}")
            stats['failed'] += 1

    print(f"\n{'='*80}")
    print(f"Preprocessing Complete!")
    print(f"{'='*80}")
    print(f"\nStatistics:")
    print(f"  Farend singletalk: {stats['farend_singletalk']}")
    if include_movement:
        print(f"  Farend singletalk (with movement): {stats['farend_singletalk_with_movement']}")
    print(f"  Too short (< 8s, skipped): {stats['too_short']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total samples created: {len(all_created)}")

    print(f"\nOutput directory: {output_dir}")

    # Verify output
    pt_files = list(output_dir.glob('*.pt'))
    print(f"\nVerification:")
    print(f"  .pt files: {len(pt_files)}")

    # Check one sample
    if pt_files:
        sample = torch.load(pt_files[0])
        print(f"\nSample shape check (first file):")
        print(f"  noisy_mic: {sample['noisy_mic'].shape}")
        print(f"  farend: {sample['farend'].shape}")
        print(f"  target: {sample['target'].shape}")
        print(f"  sample_id: {sample['sample_id']}")

    # Print storage info
    try:
        original_size = sum(f.stat().st_size for f in aec_data_dir.glob('*farend_singletalk*.wav')) / 1e9
        processed_size = sum(f.stat().st_size for f in output_dir.glob('*.pt')) / 1e9
        print(f"\nStorage:")
        print(f"  Original WAV files: {original_size:.2f} GB")
        print(f"  Processed STFT files: {processed_size:.2f} GB")
        print(f"  Size ratio: {processed_size/original_size:.2f}x")
    except:
        pass

    print(f"\nNext steps:")
    print(f"  Train model:")
    print(f"     python train.py --use_fast_loader --data_dir {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess Microsoft AEC Challenge 2023 dataset (farend singletalk, direct STFT output)'
    )
    parser.add_argument('--aec_data_dir', type=str, required=True,
                        help='Input directory with AEC Challenge WAV files')
    parser.add_argument('--output_dir', type=str, default='./data_stft',
                        help='Output directory for STFT .pt files')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT size')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Hop length')
    parser.add_argument('--win_length', type=int, default=512,
                        help='Window length')
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='Target sample rate (24kHz)')
    parser.add_argument('--chunk_sec', type=float, default=10.0,
                        help='Chunk length in seconds (10 sec default)')
    parser.add_argument('--no_movement', action='store_true',
                        help='Exclude samples with movement')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile optimization')

    args = parser.parse_args()

    preprocess_aec_farend_singletalk(
        aec_data_dir=args.aec_data_dir,
        output_dir=args.output_dir,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        sample_rate=args.sample_rate,
        chunk_length_sec=args.chunk_sec,
        include_movement=not args.no_movement,
        use_compile=not args.no_compile
    )
