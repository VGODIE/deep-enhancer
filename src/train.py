"""
Complete training script for DeepVQE
"""
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from dataset import create_dataloaders
from trainer import DeepVQETrainer
from deepvqe import DeepVQE_S
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")


def train_deepvqe(
    data_dir='./data',
    batch_size=8,  # Small batch for 10-second clips (memory limited)
    num_epochs=100,
    learning_rate=1e-4,  # Reduced from 1e-3 to prevent NaN
    num_workers=4,  # Reduce to avoid memory issues with large clips
    alpha=0.7,
    beta=0.3,
    checkpoint_dir='./checkpoints',
    log_dir='./logs',
    accelerator='auto',  # Auto-detect: cuda on RunPod, mps on Mac
    devices=1,
    use_fast_loader=True,  # Use pre-computed STFT for 10-20x speedup,
    accumulate_gradients=1
):
    """
    Train DeepVQE model

    Args:
        data_dir: Path to dataset directory
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        alpha: Weight for amplitude loss
        beta: Weight for phase loss
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        accelerator: 'auto', 'gpu', 'cpu', 'mps' (for Mac M1/M2)
        devices: Number of devices to use
    """
    print("=" * 80)
    print("DeepVQE Training")
    print("=" * 80)

    # Create dataloaders
    print("\n[1/4] Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=0.9,
        num_workers=num_workers,
        n_fft=512,
        hop_length=128,
        chunk_length_sec=10.0,  # Updated to match actual dataset (10 seconds)
        use_fast_loader=use_fast_loader  # Use pre-computed STFT if available
    )
    # Create model
    print("\n[2/4] Initializing model...")
    print(f'learning_rate: {learning_rate}')
    model = DeepVQETrainer(
        model=DeepVQE_S,
        alpha=alpha,
        beta=beta,
        power=0.3,
        lr=learning_rate
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup callbacks
    print("\n[3/4] Setting up training callbacks...")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='deepvqe-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Setup logger
    logger = TensorBoardLogger(log_dir, name='deepvqe')

    # Create trainer
    print("\n[4/4] Starting training...")

    # MPS doesn't support mixed precision well - use 32-bit for stability
    # Also detect NaN/Inf during training
    trainer = L.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        precision='32',  # Use full precision on MPS to avoid NaN issues
        gradient_clip_val=1.0,  # Clip gradients to prevent exploding gradients
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=False,
        accumulate_grad_batches=accumulate_gradients
    )
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    # Train (with checkpoint resumption support)
    # If you stop training, you can resume with the saved checkpoint
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("=" * 80)

    return trainer, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train DeepVQE model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (4 for MPS/M4 Pro, 16-32 for CUDA/RTX 5090)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for amplitude loss')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Weight for phase loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu', 'mps', 'cuda'],
                        help='Accelerator type (auto detects cuda/mps/cpu)')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices')
    parser.add_argument('--use_fast_loader', action='store_true', default=True,
                        help='Use pre-computed STFT (10-20x faster)')
    parser.add_argument('--no_fast_loader', dest='use_fast_loader', action='store_false',
                        help='Disable fast loader (compute STFT on-the-fly)')
    parser.add_argument('--accumulate_gradients', type=int, default=1)

    args = parser.parse_args()

    print('Train args: ', args)

    train_deepvqe(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        alpha=args.alpha,
        beta=args.beta,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        use_fast_loader=args.use_fast_loader,
        accumulate_gradients=args.accumulate_gradients
    )
