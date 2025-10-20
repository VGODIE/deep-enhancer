"""
Export DeepVQE models to ONNX format

Usage:
    python export_onnx.py --checkpoint ./checkpoints/best.ckpt --output model.onnx
"""
import torch
import argparse
from pathlib import Path
from deepvqe import DeepVQE_S, DeepVQE
from deepvqe_onnx import DeepVQE_S_ONNX, DeepVQE_ONNX
from onnxsim import simplify as onnx_simplify


def load_checkpoint(checkpoint_path, model_type='DeepVQE_S'):
    """Load checkpoint and create ONNX-compatible model"""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle torch.compile() wrapper
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Load into regular model first
    regular_model = DeepVQE_S() if model_type == 'DeepVQE_S' else DeepVQE()
    regular_model.load_state_dict(state_dict)
    regular_model.eval()

    # Copy to ONNX-compatible model
    onnx_model = DeepVQE_S_ONNX() if model_type == 'DeepVQE_S' else DeepVQE_ONNX()
    onnx_model.load_state_dict(regular_model.state_dict())
    onnx_model.eval()

    print(f"  Loaded {sum(p.numel() for p in onnx_model.parameters()):,} parameters")
    return onnx_model


def export_onnx(model, output_path, opset_version=17, freq_bins=257, time_steps=63):
    """Export model to ONNX format with optimizations"""
    print(f"\nExporting to ONNX: {output_path}")

    # Create dummy inputs
    dummy_mic = torch.randn(1, freq_bins, time_steps, 2)
    dummy_farend = torch.randn(1, freq_bins, time_steps, 2)

    # Export with dynamic time axis
    torch.onnx.export(
        model,
        (dummy_mic, dummy_farend),
        output_path,
        input_names=['noisy_mic', 'farend_ref'],
        output_names=['enhanced'],
        dynamic_axes={
            'noisy_mic': {2: 'time'},
            'farend_ref': {2: 'time'},
            'enhanced': {2: 'time'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )

    file_size_initial = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Exported: {file_size_initial:.2f} MB")

    # Validate if onnx is available
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  Validation: PASSED")

        # Apply ONNX optimizations
        print(f"\n  Applying optimizations...")
        try:
            import onnxoptimizer

            # Optimize with all available passes
            onnx_model = onnxoptimizer.optimize(onnx_model)
            onnx.save(onnx_model, output_path)

            file_size_opt = Path(output_path).stat().st_size / (1024 * 1024)
            reduction = ((file_size_initial - file_size_opt) / file_size_initial) * 100
            print(f"  Optimized: {file_size_opt:.2f} MB ({reduction:.1f}% reduction)")
            print(f"  Applied: BatchNorm fusion, dead code elimination, constant folding")
        except ImportError:
            print(f"  ⚠ onnxoptimizer not installed (pip install onnxoptimizer)")
            print(f"    (Model exported but not optimized)")
        except Exception as e:
            print(f"  ⚠ Optimization failed: {e}")
            print(f"    (Model exported but not optimized)")

    except ImportError:
        print(f"  Validation: SKIPPED (install onnx package)")


def main():
    parser = argparse.ArgumentParser(description='Export DeepVQE to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output path (default: ./onnx/checkpoint_name.onnx)')
    parser.add_argument('--model', type=str, default='DeepVQE_S', choices=['DeepVQE_S', 'DeepVQE'])
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem  # Get filename without extension
        args.output = f'./onnx/{checkpoint_name}.onnx'

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Export
    model = load_checkpoint(args.checkpoint, args.model)
    export_onnx(model, args.output, args.opset)

    print(f"\nDone! Use in Rust with:")
    print(f"  - ONNX Runtime: https://github.com/pykeio/ort")
    print(f"  - Input: noisy_mic(1,257,T,2), farend_ref(1,257,T,2)")
    print(f"  - Output: enhanced(1,257,T,2)")
    print(f"  - STFT: sample_rate=24000, n_fft=512, hop_length=128")


if __name__ == "__main__":
    main()
