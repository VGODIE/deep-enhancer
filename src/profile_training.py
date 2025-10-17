"""
Profile training to identify bottlenecks
Works on CUDA, MPS, and CPU
"""
import torch
import time
from dataset import create_dataloaders
from trainer import DeepVQETrainer
from deepvqe import DeepVQE_S

def sync_device(device):
    """Synchronize device for accurate timing"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

def get_memory_usage(device):
    """Get memory usage for device"""
    if device.type == 'cuda':
        return {
            'allocated': torch.cuda.memory_allocated()/1e9,
            'reserved': torch.cuda.memory_reserved()/1e9,
            'max_allocated': torch.cuda.max_memory_allocated()/1e9
        }
    elif device.type == 'mps':
        return {
            'allocated': torch.mps.current_allocated_memory()/1e9,
            'reserved': 0,  # MPS doesn't expose reserved memory
            'max_allocated': 0
        }
    return None

def profile_training_step():
    """Profile a single training step to find bottlenecks"""

    print("="*80)
    print("Training Profiler - Finding Bottlenecks")
    print("="*80)

    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    # Create small dataloader
    print("\nLoading data...")
    train_loader, _ = create_dataloaders(
        data_dir='./data',
        batch_size=1,
        num_workers=0,  # Single-threaded for profiling
        use_fast_loader=True
    )

    # Create model
    print("Creating model...")
    model = DeepVQETrainer(
        model=DeepVQE_S,
        alpha=0.7,
        beta=0.3,
        lr=1e-4
    ).to(device)

    # Get one batch
    batch = next(iter(train_loader))

    # Move to device
    batch = {
        'noisy_mic': batch['noisy_mic'].to(device),
        'farend': batch['farend'].to(device),
        'target': batch['target'].to(device),
        'sample_ids': batch['sample_ids']
    }

    print(f"\nBatch shapes:")
    print(f"  noisy_mic: {batch['noisy_mic'].shape}")
    print(f"  farend: {batch['farend'].shape}")
    print(f"  target: {batch['target'].shape}")

    # Warmup
    print("\nWarming up (3 iterations)...")
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model.training_step(batch, 0)
    sync_device(device)

    # Manual timing breakdown
    print("\n" + "="*80)
    print("Detailed Timing Breakdown (batch_size=1):")
    print("="*80)

    model.eval()
    with torch.no_grad():
        # Data extraction
        t0 = time.time()
        noisy_mic = batch['noisy_mic']
        farend = batch['farend']
        target = batch['target']
        sync_device(device)
        t1 = time.time()
        data_extract_time = (t1-t0)*1000
        print(f"1. Data extraction: {data_extract_time:.2f} ms")

        # Model forward
        t0 = time.time()
        enhanced = model.model(noisy_mic, farend)
        sync_device(device)
        t1 = time.time()
        forward_time = (t1-t0)*1000
        print(f"2. Model forward: {forward_time:.2f} ms")

        # Loss computation
        t0 = time.time()
        loss, loss_dict = model.loss_fn(enhanced, target)
        sync_device(device)
        t1 = time.time()
        loss_time = (t1-t0)*1000
        print(f"3. Loss computation: {loss_time:.2f} ms")

        total_inference = data_extract_time + forward_time + loss_time
        print(f"\nTotal (inference only): {total_inference:.2f} ms")

    # Test backward pass
    print("\n" + "="*80)
    print("With Backward Pass (training mode):")
    print("="*80)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()
        optimizer.step()
    sync_device(device)

    # Time full training step
    t0 = time.time()
    optimizer.zero_grad()
    loss = model.training_step(batch, 0)
    loss.backward()
    optimizer.step()
    sync_device(device)
    t1 = time.time()
    full_training_time = (t1-t0)*1000

    print(f"Full training step: {full_training_time:.2f} ms")
    print(f"  Forward: ~{total_inference:.2f} ms ({total_inference/full_training_time*100:.1f}%)")
    print(f"  Backward+Optimizer: ~{full_training_time-total_inference:.2f} ms ({(full_training_time-total_inference)/full_training_time*100:.1f}%)")

    # Test different batch sizes
    print("\n" + "="*80)
    print("Batch Size Scaling Analysis:")
    print("="*80)

    batch_sizes = [1, 2, 4]
    timings = []

    for bs in batch_sizes:
        try:
            loader, _ = create_dataloaders(
                data_dir='./data',
                batch_size=bs,
                num_workers=0,
                use_fast_loader=True
            )

            test_batch = next(iter(loader))
            test_batch = {
                'noisy_mic': test_batch['noisy_mic'].to(device),
                'farend': test_batch['farend'].to(device),
                'target': test_batch['target'].to(device),
                'sample_ids': test_batch['sample_ids']
            }

            model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = model.model(test_batch['noisy_mic'], test_batch['farend'])
                sync_device(device)

                # Time it
                t0 = time.time()
                for _ in range(5):
                    _ = model.model(test_batch['noisy_mic'], test_batch['farend'])
                sync_device(device)
                t1 = time.time()

                avg_time = (t1-t0)/5 * 1000
                time_per_sample = avg_time / bs
                timings.append((bs, avg_time, time_per_sample))

                print(f"Batch size {bs}: {avg_time:.2f} ms total, {time_per_sample:.2f} ms/sample")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch size {bs}: OOM Error")
                break
            else:
                raise

    # Memory usage
    mem_info = get_memory_usage(device)
    if mem_info:
        print("\n" + "="*80)
        print("Memory Usage:")
        print("="*80)
        print(f"Allocated: {mem_info['allocated']:.2f} GB")
        if mem_info['reserved'] > 0:
            print(f"Reserved: {mem_info['reserved']:.2f} GB")
        if mem_info['max_allocated'] > 0:
            print(f"Max allocated: {mem_info['max_allocated']:.2f} GB")

    # Analysis
    print("\n" + "="*80)
    print("Bottleneck Analysis:")
    print("="*80)

    if forward_time > loss_time * 3:
        print("⚠️  Model forward pass is the bottleneck")
        print("    → GRU/RNN layers are likely slowing things down")
        print("    → Consider using smaller hidden sizes or removing GRU")
    elif loss_time > forward_time:
        print("⚠️  Loss computation is slow")
        print("    → Phase loss with atan2/cos operations is expensive")
        print("    → Try disabling NaN checks or simplifying loss")
    else:
        print("✓ Forward and loss are balanced")

    if full_training_time > 1000:
        print("\n⚠️  Training is slow (>1 second per batch)")
        print("    Possible causes:")
        print("    1. Low GPU utilization (check nvidia-smi/Activity Monitor)")
        print("    2. Data loading bottleneck (try increasing num_workers)")
        print("    3. Small batch size (try gradient accumulation)")

    print("\n" + "="*80)
    print("Recommendations:")
    print("="*80)

    if timings:
        best_bs = max(timings, key=lambda x: x[0])[0]
        print(f"1. Maximum stable batch size: {best_bs}")

        if best_bs == 1:
            print("   → Use gradient accumulation to simulate larger batches:")
            print("      python src/train.py --batch_size 1 --accumulate_gradients 4")
        else:
            print(f"   → Try: python src/train.py --batch_size {best_bs}")

    print(f"\n2. For better GPU utilization:")
    print("   → Increase num_workers (try 4-8 on CPU, 2-4 on your machine)")
    print("      python src/train.py --num_workers 8")

    print(f"\n3. Current bottleneck breakdown:")
    print(f"   → Model forward: {forward_time:.1f} ms ({forward_time/total_inference*100:.1f}%)")
    print(f"   → Loss computation: {loss_time:.1f} ms ({loss_time/total_inference*100:.1f}%)")

    print("\n" + "="*80)


if __name__ == "__main__":
    profile_training_step()
