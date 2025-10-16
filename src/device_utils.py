"""
Device detection and configuration utilities for cross-platform training
Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU
"""
import os
import torch


def get_device(accelerator=None):
    """
    Auto-detect or manually set device/accelerator type

    Args:
        accelerator: Optional override ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        tuple: (accelerator_name, devices_count)

    Examples:
        >>> # Auto-detect
        >>> accelerator, devices = get_device()

        >>> # Manual override via function argument
        >>> accelerator, devices = get_device('cuda')

        >>> # Manual override via environment variable
        >>> os.environ['ACCELERATOR'] = 'cuda'
        >>> accelerator, devices = get_device()
    """
    # Check environment variable first
    env_accelerator = os.environ.get('ACCELERATOR', '').lower()
    if env_accelerator:
        accelerator = env_accelerator

    # Manual override
    if accelerator:
        accelerator = accelerator.lower()

        if accelerator == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available. Install CUDA-enabled PyTorch.")
            devices = torch.cuda.device_count()
            return 'cuda', devices

        elif accelerator == 'mps':
            if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
                raise ValueError("MPS requested but not available. Run on Apple Silicon Mac.")
            return 'mps', 1

        elif accelerator == 'cpu':
            return 'cpu', 1

        else:
            raise ValueError(f"Unknown accelerator: {accelerator}. Choose 'cuda', 'mps', or 'cpu'.")

    # Auto-detect
    if torch.cuda.is_available():
        devices = torch.cuda.device_count()
        return 'cuda', devices
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return 'mps', 1
    else:
        return 'cpu', 1


def get_precision(accelerator):
    """
    Get recommended precision for the accelerator

    Args:
        accelerator: 'cuda', 'mps', or 'cpu'

    Returns:
        str: Precision string for PyTorch Lightning ('32', '16-mixed', 'bf16-mixed')
    """
    if accelerator == 'cuda':
        # Check if GPU supports bfloat16 (RTX 30xx+ and newer)
        if torch.cuda.is_available():
            # RTX 5090 and modern GPUs support bfloat16
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere (RTX 30xx) and newer
                return 'bf16-mixed'
            else:
                return '16-mixed'  # Older GPUs use fp16
        return '16-mixed'
    elif accelerator == 'mps':
        # MPS has issues with mixed precision, use full precision
        return '32'
    else:
        return '32'


def print_device_info(accelerator, devices):
    """Print detailed device information"""
    print("\n" + "=" * 80)
    print("DEVICE CONFIGURATION")
    print("=" * 80)

    if accelerator == 'cuda':
        print(f"Accelerator: CUDA (NVIDIA GPU)")
        print(f"Available GPUs: {devices}")
        for i in range(devices):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    elif accelerator == 'mps':
        print(f"Accelerator: MPS (Apple Silicon)")
        print(f"Devices: 1 (unified memory)")
    else:
        print(f"Accelerator: CPU")
        print(f"Note: Training will be slow without GPU acceleration")

    precision = get_precision(accelerator)
    print(f"Recommended precision: {precision}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test the device detection
    accelerator, devices = get_device()
    print_device_info(accelerator, devices)

    print("\nTest manual overrides:")
    for override in ['cuda', 'mps', 'cpu']:
        try:
            acc, dev = get_device(override)
            print(f"  {override}: OK ({acc}, {dev} device(s))")
        except ValueError as e:
            print(f"  {override}: {e}")
