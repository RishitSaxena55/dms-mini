"""
Export PyTorch model to ONNX format.
"""

import argparse
import torch
import onnx
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from models.emotion_small_cnn import create_emotion_model


def export_to_onnx(model, output_path, img_size, channels, opset_version=13):
    """Export model to ONNX format."""
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, channels, img_size, img_size)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"✓ Model exported to {output_path}")
    print(f"  Input shape: [batch, {channels}, {img_size}, {img_size}]")
    print(f"  Output shape: [batch, 7]")

    # Test inference
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)

    test_input = dummy_input.numpy()
    ort_output = session.run(None, {'input': test_input})[0]

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    # Check parity
    diff = abs(torch_output - ort_output).max()
    print(f"  Max difference (PyTorch vs ONNX): {diff:.6f}")

    if diff < 1e-4:
        print("  ✓ ONNX export verified!")
    else:
        print(f"  ⚠ Warning: Large difference detected!")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX path')
    parser.add_argument('--img-size', type=int, default=64, help='Input size')
    parser.add_argument('--channels', type=int, default=1, help='Input channels (1 or 3)')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(args.weights, map_location='cpu')
    model_args = checkpoint.get('args', {})

    grayscale = model_args.get('grayscale', args.channels == 1)

    model = create_emotion_model(num_classes=7, grayscale=grayscale)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Export
    export_to_onnx(model, args.output, args.img_size, args.channels, args.opset)

    # Print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Model size: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
