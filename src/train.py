"""
Complete training script for DeepVQE
"""
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from dataset import create_dataloaders, create_dataloaders_wds
from trainer import DeepVQETrainer
from deepvqe import DeepVQE_S
from hf_hub_utils import HFCheckpointUploader
import warnings
import os
import yaml

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")

def train(
        data_mode='local',
        data_dir='./data_stft',
        train_split=0.9,
        hf_repo_id=None,
        train_tar_pattern='train-*.tar.gz',
        val_tar_pattern='val-*.tar.gz',
        cache_dir=None,
        batch_size=8,
        num_epochs=100,
        learning_rate=1e-4,
        num_workers=4,
        # Loss configuration
        loss_type='multi_resolution',
        alpha=0.7,
        beta=0.3,
        lambda_sc=1.0,
        lambda_mag=1.0,
        lambda_sisdr=0.5,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        experiment_name='default',
        checkpoint_dir='./checkpoints',
        log_dir='./logs',
        accelerator='auto',
        accumulate_gradients=1,
        hf_checkpoint_repo_id=None,
        hf_token=None):
    """
    Pure PyTorch training loop for DeepVQE

    Args:
        data_mode: 'local' or 'webdataset'
        data_dir: Path to local STFT dataset (for local mode)
        train_split: Train/val split ratio (for local mode)
        hf_repo_id: HuggingFace dataset repo ID (for webdataset mode)
        train_tar_pattern: Pattern/list for train tar files (for webdataset mode)
        val_tar_pattern: Pattern/list for val tar files (for webdataset mode)
        cache_dir: Cache dir for webdataset shards
        experiment_name: Subdirectory name for this experiment
        hf_checkpoint_repo_id: HuggingFace repo for checkpoint uploads
    """
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter
    from loss import DeepVQELoss
    from multiloss import MultiResolutionLoss
    from tqdm import tqdm

    print("=" * 80)
    print("DeepVQE Training (Pure PyTorch)")
    print("=" * 80)

    # Setup Hugging Face Hub uploader if repo_id provided
    hf_uploader = None
    if hf_checkpoint_repo_id:
        try:
            print("\n[HF Hub] Setting up checkpoint auto-upload...")
            hf_uploader = HFCheckpointUploader(
                repo_id=hf_checkpoint_repo_id,
                token=hf_token or os.getenv("HF_TOKEN"),
                private=True
            )
            print(f"[HF Hub] ✓ Checkpoints will be uploaded to: https://huggingface.co/{hf_checkpoint_repo_id}")
        except Exception as e:
            print(f"[HF Hub] ⚠ Warning: Failed to setup HF Hub uploader ({e})")
            print("[HF Hub] Continuing without auto-upload...")
            hf_uploader = None

    # Create experiment-specific directories
    checkpoint_dir = Path(checkpoint_dir) / experiment_name
    log_dir = Path(log_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {experiment_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")

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

    if data_mode == 'webdataset':
        if hf_repo_id:
            print(f"Using WebDataset streaming from HuggingFace Hub: {hf_repo_id}")
        else:
            print("Using WebDataset from local tar files")
        train_loader, val_loader = create_dataloaders_wds(
            hf_repo_id=hf_repo_id,
            train_tar_pattern=train_tar_pattern,
            val_tar_pattern=val_tar_pattern,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            hf_token=hf_token,
            cache_dir=cache_dir,
        )
    else:
        print(f"Using local dataset from: {data_dir}")
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_split=train_split,
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

    # Setup loss function based on configuration
    print(f"\n[Loss] Using loss type: {loss_type}")
    if loss_type == 'multi_resolution':
        loss_fn = MultiResolutionLoss(
            original_n_fft=512,
            original_hop_length=128,
            original_win_length=512,
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_sizes=win_sizes,
            lambda_sc=lambda_sc,
            lambda_mag=lambda_mag,
            lambda_sisdr=lambda_sisdr
        ).to(device)
        print(f"  Multi-Resolution STFT Loss:")
        print(f"    - STFT resolutions: {fft_sizes}")
        print(f"    - Weights: λ_sc={lambda_sc}, λ_mag={lambda_mag}, λ_sisdr={lambda_sisdr}")
    else:  # 'original'
        loss_fn = DeepVQELoss(alpha=alpha, beta=beta, power=0.3)
        print(f"  Original DeepVQE Loss: α={alpha}, β={beta}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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

    last_ckpt = checkpoint_dir / 'last.ckpt'

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
    writer = SummaryWriter(log_dir=str(log_dir))

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
        train_loss_components = {}  # Accumulate all loss components dynamically
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

            # Accumulate metrics dynamically
            train_loss_sum += loss.item() * accumulate_gradients
            for key, value in loss_dict.items():
                if key not in train_loss_components:
                    train_loss_components[key] = 0.0
                train_loss_components[key] += value
            train_batches += 1

            # Update progress bar with current loss (show first 2 components)
            postfix = {'loss': f"{loss.item() * accumulate_gradients:.4f}"}
            for i, (key, value) in enumerate(loss_dict.items()):
                if i < 2 and key != 'total_loss':  # Show first 2 non-total metrics
                    postfix[key.replace('_loss', '')] = f"{value:.4f}"
            train_pbar.set_postfix(postfix)

            # Log to TensorBoard every 10 steps
            if batch_idx % 10 == 0:
                writer.add_scalar('train/loss_step', loss.item() * accumulate_gradients, global_step)

        train_pbar.close()

        # Calculate epoch metrics
        train_loss_avg = train_loss_sum / train_batches
        train_loss_components_avg = {k: v / train_batches for k, v in train_loss_components.items()}

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_loss_components = {}
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
                for key, value in loss_dict.items():
                    if key not in val_loss_components:
                        val_loss_components[key] = 0.0
                    val_loss_components[key] += value
                val_batches += 1

                # Update progress bar (show first 2 components)
                postfix = {'loss': f"{loss.item():.4f}"}
                for i, (key, value) in enumerate(loss_dict.items()):
                    if i < 2 and key != 'total_loss':
                        postfix[key.replace('_loss', '')] = f"{value:.4f}"
                val_pbar.set_postfix(postfix)

        val_pbar.close()

        val_loss_avg = val_loss_sum / val_batches
        val_loss_components_avg = {k: v / val_batches for k, v in val_loss_components.items()}

        # Log epoch metrics
        writer.add_scalar('train/loss_epoch', train_loss_avg, epoch)
        for key, value in train_loss_components_avg.items():
            if key != 'total_loss':
                writer.add_scalar(f'train/{key}', value, epoch)

        writer.add_scalar('val/loss_epoch', val_loss_avg, epoch)
        for key, value in val_loss_components_avg.items():
            if key != 'total_loss':
                writer.add_scalar(f'val/{key}', value, epoch)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Update epoch progress bar with summary
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss_avg:.4f}",
            'val_loss': f"{val_loss_avg:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        # Print epoch summary
        tqdm.write(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")

        # Format train loss components
        train_components_str = ", ".join([f"{k.replace('_loss', '')}: {v:.4f}"
                                          for k, v in train_loss_components_avg.items()
                                          if k != 'total_loss'])
        tqdm.write(f"  Train Loss: {train_loss_avg:.4f} ({train_components_str})")

        # Format val loss components
        val_components_str = ", ".join([f"{k.replace('_loss', '')}: {v:.4f}"
                                        for k, v in val_loss_components_avg.items()
                                        if k != 'total_loss'])
        tqdm.write(f"  Val Loss:   {val_loss_avg:.4f} ({val_components_str})")
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
        last_ckpt_path = checkpoint_dir / 'last.ckpt'
        torch.save(checkpoint, str(last_ckpt_path))

        # Save best checkpoint
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_ckpt_path = checkpoint_dir / f'best-epoch{epoch:02d}-val{val_loss_avg:.4f}.ckpt'
            torch.save(checkpoint, str(best_ckpt_path))
            tqdm.write(f"  *** New best model saved: {best_ckpt_path}")

            # Upload best checkpoint to HF Hub
            if hf_uploader:
                hf_uploader.upload_checkpoint(
                    str(best_ckpt_path),
                    commit_message=f"Best model - Epoch {epoch+1}, Val Loss: {val_loss_avg:.4f}",
                    subfolder=f"checkpoints/{experiment_name}"
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
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Train DeepVQE model')
    parser.add_argument('--config', type=str, default='params.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--data_mode', type=str, choices=['local', 'webdataset'],
                        help='Override data_mode from config')
    parser.add_argument('--batch_size', type=int,
                        help='Override batch_size from config')
    parser.add_argument('--epochs', type=int,
                        help='Override num_epochs from config')
    parser.add_argument('--lr', type=float,
                        help='Override learning_rate from config')

    args = parser.parse_args()

    # Load config from YAML
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")

    # Override with command line arguments
    if args.data_mode:
        config['data_mode'] = args.data_mode
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr

    print("\nTraining configuration:")
    print(yaml.dump(config, default_flow_style=False))

    train(
        data_mode=config['data_mode'],
        data_dir=config['data_dir'],
        train_split=config['train_split'],
        hf_repo_id=config.get('hf_repo_id'),
        train_tar_pattern=config['train_tar_pattern'],
        val_tar_pattern=config['val_tar_pattern'],
        cache_dir=config.get('cache_dir'),
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        num_workers=config['num_workers'],
        # Loss configuration
        loss_type=config.get('loss_type', 'multi_resolution'),
        alpha=config.get('alpha', 0.7),
        beta=config.get('beta', 0.3),
        lambda_sc=config.get('lambda_sc', 1.0),
        lambda_mag=config.get('lambda_mag', 1.0),
        lambda_sisdr=config.get('lambda_sisdr', 0.5),
        fft_sizes=config.get('fft_sizes', [512, 1024, 2048]),
        hop_sizes=config.get('hop_sizes', [128, 256, 512]),
        win_sizes=config.get('win_sizes', [512, 1024, 2048]),
        experiment_name=config['experiment_name'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir'],
        accelerator=config['accelerator'],
        accumulate_gradients=config['accumulate_gradients'],
        hf_checkpoint_repo_id=config.get('hf_checkpoint_repo_id'),
        hf_token=None
    )
