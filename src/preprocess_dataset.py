"""
Pre-process dataset: Convert WAV to STFT and save to disk
This dramatically speeds up training by avoiding real-time STFT computation

Usage:
    python preprocess_dataset.py --data_dir ./data --output_dir ./data_stft

Expected speedup: 10-20x faster training!
"""
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import argparse


def preprocess_audio_to_stft(
    data_dir='./data',
    output_dir='./data_stft',
    n_fft=512,
    hop_length=128,
    win_length=512,
    sample_rate=24000,  # Downsample to 24kHz as per DeepVQE paper (saves 50% memory!)
    chunk_length_sec=10.0  # Your dataset's actual duration (10 seconds)
):
    """
    Pre-process all audio files to STFT format

    This saves STFT tensors as .pt files, avoiding expensive real-time computation
    Speed improvement: ~10-20x faster training!
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"{'='*80}")
    print(f"DeepVQE Dataset Preprocessing")
    print(f"{'='*80}")

    # STFT transform
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=None,
        return_complex=True
    )

    # Find all samples
    mic_files = sorted(data_dir.glob('*_mic.wav'))
    print(f"\nFound {len(mic_files)} samples in {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"STFT config: n_fft={n_fft}, hop={hop_length}, win={win_length}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Chunk length: {chunk_length_sec} seconds")

    chunk_samples = int(chunk_length_sec * sample_rate) if chunk_length_sec else None

    def load_and_convert(file_path):
        """Load audio and convert to STFT format"""
        waveform, sr = torchaudio.load(file_path)

        # Mono conversion
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        # Chunk to fixed length (IMPORTANT: ensures all samples same length!)
        if chunk_samples is not None:
            min_required_samples = int(chunk_samples * 0.8)  # Require at least 80% of target length (8 sec)

            if waveform.shape[0] < min_required_samples:
                # Too short - skip this sample
                return None
            elif waveform.shape[0] > chunk_samples:
                # Take center chunk (deterministic for preprocessing)
                start = (waveform.shape[0] - chunk_samples) // 2
                waveform = waveform[start:start + chunk_samples]
            elif waveform.shape[0] < chunk_samples:
                # Pad if slightly too short
                waveform = torch.nn.functional.pad(waveform, (0, chunk_samples - waveform.shape[0]))

        # Compute STFT
        stft = stft_transform(waveform)  # Complex tensor (F, T)

        # Convert to (F, T, 2) format [real, imag]
        stft_complex = torch.stack([stft.real, stft.imag], dim=-1)

        return stft_complex

    # Process all files
    print(f"\n{'='*80}")
    print("Processing files...")
    print(f"{'='*80}\n")

    failed = []
    for mic_file in tqdm(mic_files, desc="Converting WAV to STFT"):
        sample_id = mic_file.stem.replace('_mic', '')

        try:
            # Define paths
            mic_path = data_dir / f"{sample_id}_mic.wav"
            farend_path = data_dir / f"{sample_id}_farend.wav"
            target_path = data_dir / f"{sample_id}_target.wav"

            # Check all files exist
            if not all([mic_path.exists(), farend_path.exists(), target_path.exists()]):
                tqdm.write(f"Missing files for {sample_id}, skipping")
                failed.append(sample_id)
                continue

            # Convert to STFT
            mic_stft = load_and_convert(mic_path)
            farend_stft = load_and_convert(farend_path)
            target_stft = load_and_convert(target_path)

            # Check if any are too short (None returned)
            if mic_stft is None or farend_stft is None or target_stft is None:
                tqdm.write(f"Skipping {sample_id} - too short (< 8 seconds)")
                failed.append(sample_id)
                continue

            # Save as single .pt file
            output_file = output_dir / f"{sample_id}.pt"
            torch.save({
                'noisy_mic': mic_stft,
                'farend': farend_stft,
                'target': target_stft,
                'sample_id': sample_id
            }, output_file)

        except Exception as e:
            tqdm.write(f"Error processing {sample_id}: {e}")
            failed.append(sample_id)

    print(f"\n{'='*80}")
    print(f"Preprocessing Complete!")
    print(f"{'='*80}")
    print(f"Processed: {len(mic_files) - len(failed)}/{len(mic_files)} samples")
    print(f"Output directory: {output_dir}")

    if failed:
        print(f"\nFailed samples ({len(failed)}): {failed[:10]}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # Print storage info
    try:
        original_size = sum(f.stat().st_size for f in data_dir.glob('*.wav')) / 1e9
        processed_size = sum(f.stat().st_size for f in output_dir.glob('*.pt')) / 1e9
        print(f"\nStorage:")
        print(f"  Original WAV files: {original_size:.2f} GB")
        print(f"  Processed STFT files: {processed_size:.2f} GB")
        print(f"  Size ratio: {processed_size/original_size:.2f}x")
    except:
        pass

    print(f"\nNext step:")
    print(f"  Run training with: python train.py --use_fast_loader")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess audio dataset to STFT')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Input directory with WAV files')
    parser.add_argument('--output_dir', type=str, default='./data_stft',
                        help='Output directory for STFT files')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT size')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Hop length')
    parser.add_argument('--chunk_sec', type=float, default=10.0,
                        help='Chunk length in seconds (10 sec for your dataset)')

    args = parser.parse_args()

    preprocess_audio_to_stft(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        chunk_length_sec=args.chunk_sec
    )
