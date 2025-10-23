import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTToWaveform(nn.Module):
    """
    Converts complex STFT representation to waveform using inverse STFT

    Input format: (B, F, T, 2) where [..., 0]=real, [..., 1]=imag
    Output format: (B, L) waveform
    """
    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Register window as buffer (moves with model to correct device)
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, stft_complex):
        """
        Args:
            stft_complex: (B, F, T, 2) - complex STFT with real/imag in last dim

        Returns:
            waveform: (B, L) - reconstructed waveform
        """
        # Convert from (B, F, T, 2) to complex tensor (B, F, T)
        stft = torch.complex(stft_complex[..., 0], stft_complex[..., 1])

        # Inverse STFT to get waveform
        # torch.istft expects (B, F, T) complex tensor
        waveform = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False
        )

        return waveform


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss

    SC = ||magnitude(pred) - magnitude(target)||_F / ||magnitude(target)||_F

    Measures the relative error in spectral magnitude.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag: (B, F, T) - predicted magnitude
            target_mag: (B, F, T) - target magnitude
        """
        return torch.norm(pred_mag - target_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)


class LogSTFTMagnitudeLoss(nn.Module):
    """
    Log STFT Magnitude Loss

    L = ||log(magnitude(pred) + c) - log(magnitude(target) + c)||_1

    L1 loss on log-compressed magnitudes. More perceptually aligned than linear magnitudes.
    """
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_mag, target_mag):
        """
        Args:
            pred_mag: (B, F, T) - predicted magnitude
            target_mag: (B, F, T) - target magnitude
        """
        return F.l1_loss(
            torch.log(pred_mag + self.epsilon),
            torch.log(target_mag + self.epsilon)
        )


class STFTLoss(nn.Module):
    """
    Single-resolution STFT Loss

    Combines spectral convergence and log magnitude loss.
    """
    def __init__(self, fft_size=1024, hop_size=256, win_size=1024, epsilon=1e-5):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.epsilon = epsilon

        self.spectral_convergence = SpectralConvergenceLoss()
        self.log_stft_magnitude = LogSTFTMagnitudeLoss(epsilon)

        # Register window as buffer (moves with model to correct device)
        window = torch.hann_window(win_size)
        self.register_buffer('window', window)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, L) - predicted waveform
            target: (B, L) - target waveform

        Returns:
            sc_loss: spectral convergence loss
            mag_loss: log magnitude loss
        """
        # Compute STFT
        pred_stft = torch.stft(
            pred,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True
        )
        target_stft = torch.stft(
            target,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True
        )

        # Compute magnitudes
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        # Compute losses
        sc_loss = self.spectral_convergence(pred_mag, target_mag)
        mag_loss = self.log_stft_magnitude(pred_mag, target_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss

    Computes STFT loss at multiple resolutions to capture features at different time scales.

    Common configurations:
        - [512, 1024, 2048] - Fast, good for speech
        - [512, 1024, 2048, 4096] - More thorough

    Used in: Parallel WaveGAN, HiFi-GAN, DEMUCS
    """
    def __init__(self,
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[128, 256, 512],
                 win_sizes=[512, 1024, 2048],
                 epsilon=1e-5):
        """
        Args:
            fft_sizes: List of FFT sizes
            hop_sizes: List of hop sizes (typically fft_size // 4)
            win_sizes: List of window sizes (typically same as fft_size)
            epsilon: Small constant for numerical stability
        """
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes), \
            "Number of FFT sizes, hop sizes, and window sizes must match"

        self.stft_losses = nn.ModuleList()
        for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses.append(
                STFTLoss(fft_size, hop_size, win_size, epsilon)
            )

    def forward(self, pred, target):
        """
        Args:
            pred: (B, L) - predicted waveform
            target: (B, L) - target waveform

        Returns:
            sc_loss: total spectral convergence loss (averaged across resolutions)
            mag_loss: total log magnitude loss (averaged across resolutions)
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for stft_loss in self.stft_losses:
            sc_l, mag_l = stft_loss(pred, target)
            sc_loss += sc_l
            mag_loss += mag_l

        # Average across resolutions
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class SI_SDR_Loss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss

    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    where:
        s_target = <s, s_hat> / ||s_hat||^2 * s_hat  (scaled target projection)
        e_noise = s - s_target                        (residual error)
        s = predicted signal
        s_hat = target signal

    Scale-invariant: Insensitive to amplitude differences between pred and target.
    Widely used in speech enhancement/separation.

    Returns negative SI-SDR as loss (higher SI-SDR is better, so we minimize -SI-SDR).
    """
    def __init__(self, zero_mean=True, eps=1e-8):
        """
        Args:
            zero_mean: If True, remove mean from signals before computing SI-SDR
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: (B, L) - predicted waveform
            target: (B, L) - target waveform

        Returns:
            loss: negative SI-SDR (scalar)
        """
        # Remove mean if specified (common practice)
        if self.zero_mean:
            pred = pred - pred.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)

        # Compute scaling factor: <s, s_hat> / ||s_hat||^2
        # Shape: (B,)
        dot_product = torch.sum(pred * target, dim=-1)
        target_energy = torch.sum(target ** 2, dim=-1) + self.eps
        scale = dot_product / target_energy

        # Compute scaled target projection: s_target = scale * s_hat
        # Shape: (B, L)
        s_target = scale.unsqueeze(-1) * target

        # Compute error: e_noise = s - s_target
        e_noise = pred - s_target

        # Compute SI-SDR: 10 * log10(||s_target||^2 / ||e_noise||^2)
        s_target_energy = torch.sum(s_target ** 2, dim=-1) + self.eps
        e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + self.eps

        si_sdr = 10 * torch.log10(s_target_energy / e_noise_energy)

        # Return negative SI-SDR as loss (we want to maximize SI-SDR, so minimize -SI-SDR)
        # Average over batch
        return -si_sdr.mean()


class MultiResolutionLoss(nn.Module):
    """
    Combined Multi-Resolution STFT + SI-SDR Loss for STFT-domain inputs

    This loss function:
    1. Converts STFT (B, F, T, 2) to waveform via iSTFT
    2. Computes Multi-Resolution STFT loss at multiple resolutions
    3. Computes SI-SDR loss in time domain

    Total Loss = λ_sc * SC_loss + λ_mag * Mag_loss + λ_sisdr * SI-SDR_loss

    This is the state-of-the-art loss for speech enhancement and vocoding tasks.

    Typical configurations:
        - Speech enhancement: lambda_sc=1.0, lambda_mag=1.0, lambda_sisdr=0.5
        - Vocoder (HiFi-GAN): lambda_sc=1.0, lambda_mag=1.0, lambda_sisdr=0.1
        - Music separation: lambda_sc=1.0, lambda_mag=1.0, lambda_sisdr=1.0
    """
    def __init__(self,
                 # iSTFT parameters (must match your preprocessing)
                 original_n_fft=512,
                 original_hop_length=128,
                 original_win_length=512,
                 # Multi-resolution STFT parameters
                 fft_sizes=[512, 1024, 2048],
                 hop_sizes=[128, 256, 512],
                 win_sizes=[512, 1024, 2048],
                 # Loss weights
                 lambda_sc=1.0,
                 lambda_mag=1.0,
                 lambda_sisdr=0.5,
                 epsilon=1e-5,
                 zero_mean_sisdr=True):
        """
        Args:
            n_fft: FFT size for iSTFT (must match preprocessing)
            hop_length: Hop length for iSTFT (must match preprocessing)
            win_length: Window length for iSTFT (must match preprocessing)
            fft_sizes: List of FFT sizes for multi-resolution STFT
            hop_sizes: List of hop sizes
            win_sizes: List of window sizes
            lambda_sc: Weight for spectral convergence loss
            lambda_mag: Weight for log magnitude loss
            lambda_sisdr: Weight for SI-SDR loss
            epsilon: Small constant for numerical stability
            zero_mean_sisdr: Whether to remove mean before computing SI-SDR
        """
        super().__init__()

        self.lambda_sc = lambda_sc
        self.lambda_mag = lambda_mag
        self.lambda_sisdr = lambda_sisdr

        # STFT to waveform conversion
        self.stft_to_wav = STFTToWaveform(original_n_fft, original_hop_length, original_win_length)

        # Multi-resolution STFT loss
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_sizes=win_sizes,
            epsilon=epsilon
        )

        # SI-SDR loss
        self.sisdr_loss = SI_SDR_Loss(zero_mean=zero_mean_sisdr, eps=epsilon)

    def forward(self, pred_stft, target_stft):
        """
        Args:
            pred_stft: (B, F, T, 2) - predicted complex STFT [real, imag]
            target_stft: (B, F, T, 2) - target complex STFT [real, imag]

        Returns:
            total_loss: combined loss (scalar)
            loss_dict: dict with individual losses for logging
        """
        # Convert STFT to waveform
        pred_wav = self.stft_to_wav(pred_stft)
        target_wav = self.stft_to_wav(target_stft)

        # Compute multi-resolution STFT loss
        sc_loss, mag_loss = self.stft_loss(pred_wav, target_wav)

        # Compute SI-SDR loss
        sisdr_loss = self.sisdr_loss(pred_wav, target_wav)

        # Combine losses
        total_loss = (
            self.lambda_sc * sc_loss +
            self.lambda_mag * mag_loss +
            self.lambda_sisdr * sisdr_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'sc_loss': sc_loss.item(),
            'mag_loss': mag_loss.item(),
            'sisdr_loss': sisdr_loss.item(),
            'si_sdr': -sisdr_loss.item()  # Positive SI-SDR for logging (dB)
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Multi-Resolution Loss with STFT inputs")
    print("=" * 80)

    # Simulate STFT data (B, F, T, 2) format
    # F = n_fft // 2 + 1 = 512 // 2 + 1 = 257
    batch_size = 4
    n_freq = 257  # n_fft // 2 + 1
    n_frames = 100

    print(f"Creating dummy STFT tensors...")
    print(f"  Shape: (B={batch_size}, F={n_freq}, T={n_frames}, 2)")

    pred_stft = torch.randn(batch_size, n_freq, n_frames, 2)
    target_stft = torch.randn(batch_size, n_freq, n_frames, 2)

    # Test STFT to Waveform conversion
    print("\n[1] Testing STFT → Waveform conversion")
    stft_to_wav = STFTToWaveform(n_fft=512, hop_length=128, win_length=512)
    pred_wav = stft_to_wav(pred_stft)
    target_wav = stft_to_wav(target_stft)
    print(f"  Input STFT shape: {pred_stft.shape}")
    print(f"  Output waveform shape: {pred_wav.shape}")

    # Test Multi-Resolution STFT Loss
    print("\n[2] Testing Multi-Resolution STFT Loss")
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048]
    )
    sc_loss, mag_loss = stft_loss(pred_wav, target_wav)
    print(f"  Spectral Convergence Loss: {sc_loss.item():.4f}")
    print(f"  Log Magnitude Loss: {mag_loss.item():.4f}")

    # Test SI-SDR Loss
    print("\n[3] Testing SI-SDR Loss")
    sisdr_loss_fn = SI_SDR_Loss()
    sisdr_loss = sisdr_loss_fn(pred_wav, target_wav)
    print(f"  SI-SDR Loss: {sisdr_loss.item():.4f}")
    print(f"  SI-SDR (positive): {-sisdr_loss.item():.4f} dB")

    # Test Combined Loss
    print("\n[4] Testing Combined Multi-Resolution Loss (STFT input)")
    combined_loss = MultiResolutionLoss(
        n_fft=512,
        hop_length=128,
        win_length=512,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        lambda_sc=1.0,
        lambda_mag=1.0,
        lambda_sisdr=0.5
    )
    total_loss, loss_dict = combined_loss(pred_stft, target_stft)
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"    - {key}: {value:.4f}")

    # Test with perfect prediction (SI-SDR should be very high)
    print("\n[5] Testing with perfect prediction")
    perfect_pred_stft = target_stft.clone()
    total_loss_perfect, loss_dict_perfect = combined_loss(perfect_pred_stft, target_stft)
    print(f"  Total Loss: {total_loss_perfect.item():.4f}")
    print(f"  SI-SDR (should be very high): {loss_dict_perfect['si_sdr']:.4f} dB")

    # Test GPU compatibility (if available)
    if torch.cuda.is_available():
        print("\n[6] Testing GPU compatibility")
        device = torch.device('cuda')
        combined_loss_gpu = combined_loss.to(device)
        pred_stft_gpu = pred_stft.to(device)
        target_stft_gpu = target_stft.to(device)
        total_loss_gpu, _ = combined_loss_gpu(pred_stft_gpu, target_stft_gpu)
        print(f"  GPU Total Loss: {total_loss_gpu.item():.4f}")
        print(f"  ✓ GPU test passed!")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
