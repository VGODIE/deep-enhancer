"""
Validate ONNX export by comparing outputs with PyTorch model

Usage:
    python validate_onnx.py --checkpoint ./checkpoints/best.ckpt --onnx ./onnx/best.onnx
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
from deepvqe import DeepVQE_S, DeepVQE


def load_pytorch_model(checkpoint_path, model_type='DeepVQE_S'):
    """Load PyTorch model from checkpoint"""
    print(f"Loading PyTorch model: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Handle torch.compile() wrapper
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model = DeepVQE_S() if model_type == 'DeepVQE_S' else DeepVQE()
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_onnx_model(onnx_path):
    """Load ONNX model"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime required (pip install onnxruntime)")
        sys.exit(1)

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # Print model info
    print(f"  Inputs: {[(i.name, i.shape) for i in session.get_inputs()]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in session.get_outputs()]}")

    return session


def validate_models(pytorch_model, onnx_session, num_tests=3):
    """Compare PyTorch and ONNX outputs on random inputs"""
    print(f"\nRunning {num_tests} validation tests...")

    all_passed = True
    time_steps_list = [63, 127, 188]

    for i, time_steps in enumerate(time_steps_list[:num_tests]):
        # Create random inputs
        mic_input = torch.randn(1, 257, time_steps, 2)
        farend_input = torch.randn(1, 257, time_steps, 2)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(mic_input, farend_input)

        # ONNX inference
        onnx_inputs = {
            'noisy_mic': mic_input.numpy(),
            'farend_ref': farend_input.numpy()
        }
        onnx_output = onnx_session.run(None, onnx_inputs)[0]

        # Compare
        pytorch_np = pytorch_output.detach().cpu().numpy()
        max_diff = np.max(np.abs(pytorch_np - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_np - onnx_output))

        print(f"\nTest {i+1} (T={time_steps}):")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        if max_diff < 1e-5:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED (diff > 1e-5)")
            all_passed = False

    return all_passed


def benchmark(pytorch_model, onnx_session, num_runs=50):
    """Compare inference speed"""
    import time

    print(f"\nBenchmarking ({num_runs} runs)...")

    # Test input (1 second of audio)
    mic_input = torch.randn(1, 257, 188, 2)
    farend_input = torch.randn(1, 257, 188, 2)

    # PyTorch warmup & benchmark
    with torch.no_grad():
        for _ in range(5):
            _ = pytorch_model(mic_input, farend_input)

        torch_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = pytorch_model(mic_input, farend_input)
            torch_times.append(time.perf_counter() - start)

    # ONNX warmup & benchmark
    onnx_inputs = {'noisy_mic': mic_input.numpy(), 'farend_ref': farend_input.numpy()}
    for _ in range(5):
        _ = onnx_session.run(None, onnx_inputs)

    onnx_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = onnx_session.run(None, onnx_inputs)
        onnx_times.append(time.perf_counter() - start)

    # Results
    torch_mean = np.mean(torch_times) * 1000
    onnx_mean = np.mean(onnx_times) * 1000

    print(f"\nPerformance (1s of audio):")
    print(f"  PyTorch: {torch_mean:.2f} ms")
    print(f"  ONNX:    {onnx_mean:.2f} ms")
    print(f"  Speedup: {torch_mean / onnx_mean:.2f}x")

    # Real-time factor
    onnx_rtf = 1000 / onnx_mean
    print(f"\n  ONNX Real-time factor: {onnx_rtf:.1f}x")
    if onnx_rtf > 1.0:
        print(f"  ✓ Suitable for real-time processing!")


def main():
    parser = argparse.ArgumentParser(description='Validate ONNX export')
    parser.add_argument('--checkpoint', type=str, required=True, help='PyTorch checkpoint')
    parser.add_argument('--onnx', type=str, required=True, help='ONNX model')
    parser.add_argument('--model', type=str, default='DeepVQE_S', choices=['DeepVQE_S', 'DeepVQE'])
    parser.add_argument('--benchmark', action='store_true', help='Run performance comparison')
    args = parser.parse_args()

    print("=" * 60)
    print("ONNX Validation")
    print("=" * 60)

    # Load models
    pytorch_model = load_pytorch_model(args.checkpoint, args.model)
    onnx_session = load_onnx_model(args.onnx)

    # Validate
    passed = validate_models(pytorch_model, onnx_session)

    # Benchmark
    if args.benchmark:
        benchmark(pytorch_model, onnx_session)

    # Summary
    print("\n" + "=" * 60)
    if passed:
        print("✓ Validation PASSED - Models produce identical outputs")
        return 0
    else:
        print("✗ Validation FAILED - Check differences above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
