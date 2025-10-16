from loss import DeepVQELoss
import lightning as L
import torch
from typing import Union, Type
import torch.nn as nn


class DeepVQETrainer(L.LightningModule):
    """
    PyTorch Lightning module for training DeepVQE

    Expected batch format:
        - noisy_mic: (B, F, T, 2) - noisy microphone signal (complex STFT)
        - system_ref: (B, F, T, 2) - system reference signal (complex STFT)
        - clean_target: (B, F, T, 2) - clean target signal (complex STFT)
    """
    def __init__(
        self,
        model: Union[nn.Module, Type[nn.Module]],
        alpha: float = 0.7,
        beta: float = 0.3,
        power: float = 2,
        lr: float = 1e-3
    ):
        """
        Args:
            model: DeepVQE model class or instance (e.g., DeepVQE or DeepVQE())
            alpha: weight for amplitude loss (default: 0.7)
            beta: weight for phase loss (default: 0.3)
            power: compression power for amplitude (default: 0.5)
            lr: learning rate (default: 1e-3)
        """
        super().__init__()
        # Support both model class and instance
        if isinstance(model, type):
            self.model = model()  # Instantiate if class is passed
        else:
            self.model = model  # Use instance directly
        self.loss_fn = DeepVQELoss(alpha=alpha, beta=beta, power=power)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, noisy_mic, system_ref):
        """Forward pass through the model"""
        return self.model(noisy_mic, system_ref)

    def training_step(self, batch, batch_idx):
        """
        Training step

        Args:
            batch: tuple of (noisy_mic, system_ref, clean_target)
                - noisy_mic: (B, F, T, 2)
                - system_ref: (B, F, T, 2)
                - clean_target: (B, F, T, 2)
        """
        noisy_mic, system_ref, clean_target = batch['noisy_mic'], batch['farend'], batch['target']

        # Validate inputs for NaN/Inf (helps debug data issues)
        if torch.isnan(noisy_mic).any() or torch.isinf(noisy_mic).any():
            raise ValueError(f"NaN/Inf detected in noisy_mic at batch {batch_idx}")
        if torch.isnan(system_ref).any() or torch.isinf(system_ref).any():
            raise ValueError(f"NaN/Inf detected in system_ref at batch {batch_idx}")
        if torch.isnan(clean_target).any() or torch.isinf(clean_target).any():
            raise ValueError(f"NaN/Inf detected in clean_target at batch {batch_idx}")

        # Forward pass
        enhanced = self.model(noisy_mic, system_ref)

        # Check model output
        if torch.isnan(enhanced).any() or torch.isinf(enhanced).any():
            raise ValueError(f"NaN/Inf detected in model output at batch {batch_idx}")

        # Compute loss
        loss, loss_dict = self.loss_fn(enhanced, clean_target)

        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"NaN/Inf detected in loss at batch {batch_idx}")

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_amp_loss', loss_dict['amp_loss'], on_step=False, on_epoch=True)
        self.log('train_phase_loss', loss_dict['phase_loss'], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step

        Args:
            batch: tuple of (noisy_mic, system_ref, clean_target)
        """
        noisy_mic, system_ref, clean_target = batch['noisy_mic'], batch['farend'], batch['target']

        # Forward pass
        enhanced = self.model(noisy_mic, system_ref)

        # Compute loss
        loss, loss_dict = self.loss_fn(enhanced, clean_target)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_amp_loss', loss_dict['amp_loss'], on_step=False, on_epoch=True)
        self.log('val_phase_loss', loss_dict['phase_loss'], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
