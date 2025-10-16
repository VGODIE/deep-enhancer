import torch
import torch.nn as nn


class CompressedAmplitudeSpectralLoss(torch.nn.Module):
    """
    Compressed amplitude spectral loss
    Often used in practice with power compression (e.g., square root)
    
    E^(amp)_{t,n} = (1/2) * (Â^p_{t,n} - A^p_{t,n})²
    
    where p is the compression power (e.g., 0.5 for square root)
    """
    def __init__(self, power=0.5):
        """
        Args:
            power: float - compression power (0.5 = square root, 1.0 = no compression)
        """
        super().__init__()
        self.power = power
    
    def forward(self, pred_complex, target_complex):
        """
        Args:
            pred_complex: (B, F, T, 2)
            target_complex: (B, F, T, 2)
        
        Returns:
            loss: scalar
        """
        # Compute amplitudes
        pred_amp = torch.sqrt(
            pred_complex[..., 0]**2 + pred_complex[..., 1]**2 + 1e-8
        )
        target_amp = torch.sqrt(
            target_complex[..., 0]**2 + target_complex[..., 1]**2 + 1e-8
        )
        
        # Apply power compression
        pred_amp_compressed = pred_amp ** self.power
        target_amp_compressed = target_amp ** self.power
        
        # MSE loss on compressed amplitudes
        loss = 0.5 * torch.mean((target_amp_compressed - pred_amp_compressed) ** 2)
        
        return loss
    

class PhaseSpectralLoss(nn.Module):
    """
    Phase spectral loss (Equation 10 in paper)
    
    E^(ph)_{t,n} = (1/2) * |1 - exp(i(θ̂_{t,n} - θ_{t,n}))|²
                 = 1 - cos(θ̂_{t,n} - θ_{t,n})
    
    This formulation handles the circular/periodic nature of phase (wraps at 2π)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_complex, target_complex):
        """
        Args:
            pred_complex: (B, F, T, 2) - predicted complex spectrum
            target_complex: (B, F, T, 2) - target complex spectrum

        Returns:
            loss: scalar
        """
        eps = 1e-8

        # Compute magnitudes for weighting
        # Phase is meaningless when magnitude is near zero, so we weight by magnitude
        pred_mag = torch.sqrt(pred_complex[..., 0]**2 + pred_complex[..., 1]**2 + eps)
        target_mag = torch.sqrt(target_complex[..., 0]**2 + target_complex[..., 1]**2 + eps)

        # Use minimum magnitude as weight (conservative approach)
        mag_weight = torch.minimum(pred_mag, target_mag)

        # Normalize magnitude to [0, 1] for numerical stability
        mag_weight = mag_weight / (torch.max(mag_weight) + eps)

        # Extract phases using atan2
        # Shapes: (B, F, T)
        pred_phase = torch.atan2(pred_complex[..., 1], pred_complex[..., 0] + eps)
        target_phase = torch.atan2(target_complex[..., 1], target_complex[..., 0] + eps)

        # Phase difference
        # Shape: (B, F, T)
        phase_diff = target_phase - pred_phase

        # Phase loss: (1/2) * |1 - exp(i*phase_diff)|²
        # This simplifies to: 1 - cos(phase_diff)
        # Weighted by magnitude to ignore phase when magnitude is small
        # Shape: (B, F, T) -> scalar
        phase_cost = (1 - torch.cos(phase_diff)) * mag_weight
        loss = torch.mean(phase_cost)

        return loss
    
    def forward_alternative(self, pred_complex, target_complex):
        """
        Alternative implementation using complex exponentials (Equation 10 directly)
        This is mathematically equivalent but more explicit
        """
        # Convert to complex tensors
        pred = torch.complex(pred_complex[..., 0], pred_complex[..., 1])
        target = torch.complex(target_complex[..., 0], target_complex[..., 1])
        
        # Get unit complex values (normalized phase): exp(iθ)
        pred_unit = pred / (torch.abs(pred) + 1e-8)
        target_unit = target / (torch.abs(target) + 1e-8)
        
        # Phase difference as complex: exp(i(θ̂ - θ))
        phase_diff_complex = target_unit * torch.conj(pred_unit)
        
        # Loss: (1/2) * |1 - exp(i(θ̂ - θ))|²
        loss = 0.5 * torch.mean(torch.abs(1 - phase_diff_complex) ** 2)

        return loss


class DeepVQELoss(nn.Module):
    """
    Combined loss for DeepVQE: amplitude + phase spectral loss

    This is commonly used in speech enhancement models.
    Total loss = α * amplitude_loss + β * phase_loss

    Default weights: α=0.7, β=0.3 (amplitude is typically weighted higher)
    """
    def __init__(self, alpha=0.7, beta=0.3, power=0.5):
        """
        Args:
            alpha: weight for amplitude loss (default: 0.7)
            beta: weight for phase loss (default: 0.3)
            power: compression power for amplitude (default: 0.5 = square root)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.amp_loss = CompressedAmplitudeSpectralLoss(power=power)
        self.phase_loss = PhaseSpectralLoss()

    def forward(self, pred_complex, target_complex):
        """
        Args:
            pred_complex: (B, F, T, 2) - predicted enhanced complex spectrum
            target_complex: (B, F, T, 2) - target clean complex spectrum

        Returns:
            loss: scalar
            loss_dict: dict with individual losses for logging
        """
        amp_loss = self.amp_loss(pred_complex, target_complex)
        phase_loss = self.phase_loss(pred_complex, target_complex)

        total_loss = self.alpha * amp_loss + self.beta * phase_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'amp_loss': amp_loss.item(),
            'phase_loss': phase_loss.item()
        }

        return total_loss, loss_dict

