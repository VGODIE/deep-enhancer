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
from hf_hub_utils import HFCheckpointUploader
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")

def train(
        data_dir='./data_stft',
        batch_size=8,
        num_epochs=100,
        learning_rate=1e-4,
        num_workers=4,
        alpha=0.7,
        beta=0.3,
        checkpoint_dir='./checkpoints',
        log_dir='./logs',
        accelerator='auto',
        accumulate_gradients=1,
        hf_repo_id=None,
        hf_token=None):
    """
    Pure PyTorch training loop for DeepVQE (without Lightning)

    Args:
        data_dir: Path to pre-computed STFT dataset directory
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        alpha: Weight for amplitude loss
        beta: Weight for phase loss
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        accelerator: 'auto', 'gpu', 'cpu', 'mps', 'cuda'
        devices: Number of devices to use (only 1 supported in pure PyTorch version)
        accumulate_gradients: Accumulate gradients over N batches
        hf_repo_id: Hugging Face repo ID (e.g., 'username/deep-enhancer-checkpoints')
        hf_token: Hugging Face token (or set HF_TOKEN environment variable)
    """
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter
    from loss import DeepVQELoss
    from tqdm import tqdm

    print("=" * 80)
    print("DeepVQE Training (Pure PyTorch)")
    print("=" * 80)

    # Setup Hugging Face Hub uploader if repo_id provided
    hf_uploader = None
    if hf_repo_id:
        try:
            print("\n[HF Hub] Setting up checkpoint auto-upload...")
            hf_uploader = HFCheckpointUploader(
                repo_id=hf_repo_id,
                token=hf_token or os.getenv("HF_TOKEN"),
                private=True
            )
            print(f"[HF Hub] ✓ Checkpoints will be uploaded to: https://huggingface.co/{hf_repo_id}")
        except Exception as e:
            print(f"[HF Hub] ⚠ Warning: Failed to setup HF Hub uploader ({e})")
            print("[HF Hub] Continuing without auto-upload...")
            hf_uploader = None

    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Determine device
    print("\n[1/5] Setting up device...")
    if accelerator == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    elif accelerator in ['cuda', 'gpu']:
        device = torch.device('cuda')
        print("Using CUDA")
    elif accelerator == 'mps':
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Create dataloaders
    print("\n[2/5] Loading dataset...")
    pin_memory = device.type == 'cuda'
    if pin_memory:
        print("Enabling pin_memory for faster CUDA transfers")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=0.9,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Create model
    print("\n[3/5] Initializing model...")
    print(f'learning_rate: {learning_rate}')
    model = DeepVQE_S().to(device)

    try:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='default')
        print("✓ Model compiled successfully")
    except Exception as e:
        print(f"⚠ Warning: torch.compile() failed ({e}). Continuing without compilation...")

    loss_fn = DeepVQELoss(alpha=alpha, beta=beta, power=0.3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Setup gradient scaler for mixed precision (CUDA only)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Using bf16-mixed precision for CUDA (saves ~50% VRAM)")
    else:
        print("Using 32-bit precision")

    # Load checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    checkpoint_path = Path(checkpoint_dir)
    last_ckpt = checkpoint_path / 'last.ckpt'

    if last_ckpt.exists():
        print(f"\n📂 Loading checkpoint: {last_ckpt}")
        try:
            checkpoint = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            global_step = checkpoint.get('global_step', 0)
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
            print(f"  Best val loss: {best_val_loss:.4f}")
            print(f"  Global step: {global_step}")
        except Exception as e:
            print(f"⚠ Warning: Failed to load checkpoint ({e}). Starting from scratch...")
            start_epoch = 0
            best_val_loss = float('inf')
            global_step = 0
    else:
        print("\n⚠ No checkpoint found, starting training from scratch")

    # Setup TensorBoard logger
    print("\n[4/5] Setting up logging...")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'deepvqe'))

    # Training state
    patience = 10

    print("\n[5/5] Starting training...")
    print("=" * 80)

    # Epoch progress bar
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch", initial=start_epoch, total=num_epochs)

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_amp_loss_sum = 0.0
        train_phase_loss_sum = 0.0
        train_batches = 0

        optimizer.zero_grad()

        # Training batch progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                         leave=False, unit="batch")

        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            noisy_mic = batch['noisy_mic'].to(device, non_blocking=True)
            system_ref = batch['farend'].to(device, non_blocking=True)
            clean_target = batch['target'].to(device, non_blocking=True)

            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    enhanced = model(noisy_mic, system_ref)
                    loss, loss_dict = loss_fn(enhanced, clean_target)
                    loss = loss / accumulate_gradients
            else:
                enhanced = model(noisy_mic, system_ref)
                loss, loss_dict = loss_fn(enhanced, clean_target)
                loss = loss / accumulate_gradients

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulate_gradients == 0:
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                global_step += 1

            # Accumulate metrics
            train_loss_sum += loss.item() * accumulate_gradients
            train_amp_loss_sum += loss_dict['amp_loss']
            train_phase_loss_sum += loss_dict['phase_loss']
            train_batches += 1

            # Update progress bar with current loss
            train_pbar.set_postfix({
                'loss': f"{loss.item() * accumulate_gradients:.4f}",
                'amp': f"{loss_dict['amp_loss']:.4f}",
                'phase': f"{loss_dict['phase_loss']:.4f}"
            })

            # Log to TensorBoard every 10 steps
            if batch_idx % 10 == 0:
                writer.add_scalar('train/loss_step', loss.item() * accumulate_gradients, global_step)

        train_pbar.close()

        # Calculate epoch metrics
        train_loss_avg = train_loss_sum / train_batches
        train_amp_loss_avg = train_amp_loss_sum / train_batches
        train_phase_loss_avg = train_phase_loss_sum / train_batches

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_amp_loss_sum = 0.0
        val_phase_loss_sum = 0.0
        val_batches = 0

        # Validation batch progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                       leave=False, unit="batch")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                noisy_mic = batch['noisy_mic'].to(device, non_blocking=True)
                system_ref = batch['farend'].to(device, non_blocking=True)
                clean_target = batch['target'].to(device, non_blocking=True)

                # Forward pass
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        enhanced = model(noisy_mic, system_ref)
                        loss, loss_dict = loss_fn(enhanced, clean_target)
                else:
                    enhanced = model(noisy_mic, system_ref)
                    loss, loss_dict = loss_fn(enhanced, clean_target)

                val_loss_sum += loss.item()
                val_amp_loss_sum += loss_dict['amp_loss']
                val_phase_loss_sum += loss_dict['phase_loss']
                val_batches += 1

                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'amp': f"{loss_dict['amp_loss']:.4f}",
                    'phase': f"{loss_dict['phase_loss']:.4f}"
                })

        val_pbar.close()

        val_loss_avg = val_loss_sum / val_batches
        val_amp_loss_avg = val_amp_loss_sum / val_batches
        val_phase_loss_avg = val_phase_loss_sum / val_batches

        # Log epoch metrics
        writer.add_scalar('train/loss_epoch', train_loss_avg, epoch)
        writer.add_scalar('train/amp_loss', train_amp_loss_avg, epoch)
        writer.add_scalar('train/phase_loss', train_phase_loss_avg, epoch)
        writer.add_scalar('val/loss_epoch', val_loss_avg, epoch)
        writer.add_scalar('val/amp_loss', val_amp_loss_avg, epoch)
        writer.add_scalar('val/phase_loss', val_phase_loss_avg, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Update epoch progress bar with summary
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss_avg:.4f}",
            'val_loss': f"{val_loss_avg:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        # Print epoch summary
        tqdm.write(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        tqdm.write(f"  Train Loss: {train_loss_avg:.4f} (Amp: {train_amp_loss_avg:.4f}, Phase: {train_phase_loss_avg:.4f})")
        tqdm.write(f"  Val Loss:   {val_loss_avg:.4f} (Amp: {val_amp_loss_avg:.4f}, Phase: {val_phase_loss_avg:.4f})")
        tqdm.write(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Learning rate scheduler step
        scheduler.step(val_loss_avg)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss_avg,
            'train_loss': train_loss_avg,
            'best_val_loss': best_val_loss,
            'global_step': global_step,
        }

        # Save last checkpoint
        last_ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')
        torch.save(checkpoint, last_ckpt_path)

        # Save best checkpoint
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_ckpt_path = os.path.join(checkpoint_dir, f'best-epoch{epoch:02d}-val{val_loss_avg:.4f}.ckpt')
            torch.save(checkpoint, best_ckpt_path)
            tqdm.write(f"  *** New best model saved: {best_ckpt_path}")

            # Upload best checkpoint to HF Hub
            if hf_uploader:
                hf_uploader.upload_checkpoint(
                    best_ckpt_path,
                    commit_message=f"Best model - Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f}"
                )

            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            tqdm.write(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break

    epoch_pbar.close()

    writer.close()

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("=" * 80)

    return model



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train DeepVQE model')
    parser.add_argument('--data_dir', type=str, default='./data_stft',
                        help='Path to pre-computed STFT dataset directory')
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
    parser.add_argument('--accumulate_gradients', type=int, default=1,
                        help='Accumulate gradients over N batches for larger effective batch size')
    parser.add_argument('--hf_repo_id', type=str, default=None,
                        help='Hugging Face repo ID for auto-uploading checkpoints (e.g., "username/deep-enhancer-ckpts"). Token from HF_TOKEN env var.')

    args = parser.parse_args()

    print('Train args: ', args)


    train(
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
            accumulate_gradients=args.accumulate_gradients,
            hf_repo_id=args.hf_repo_id,
            hf_token=None
        )
