"""
Convert ONNX to TFLite and apply INT8 quantization.
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data.datamodule import get_representative_dataset


def onnx_to_tflite(onnx_path, output_path, calib_data_dir, img_size=64, grayscale=False, quantize_int8=True):
    """Convert ONNX to TFLite with optional INT8 quantization."""

    try:
        import onnx
        from onnx_tf.backend import prepare

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('temp_saved_model')

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('temp_saved_model')

        if quantize_int8:
            print("Applying INT8 post-training quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Representative dataset
            representative_gen = get_representative_dataset(calib_data_dir, img_size, grayscale)
            converter.representative_dataset = representative_gen

            # Force full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✓ TFLite model saved to {output_path}")

        # Size
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")

        # Clean up
        import shutil
        shutil.rmtree('temp_saved_model', ignore_errors=True)

    except ImportError as e:
        print(f"Error: {e}")
        print("Install required packages: pip install onnx onnx-tf")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TFLite with INT8 quantization')
    parser.add_argument('--model', type=str, required=True, help='Input ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Output TFLite model')
    parser.add_argument('--calib-data', type=str, required=True,
                        help='Calibration data directory')
    parser.add_argument('--img-size', type=int, default=64, help='Input image size')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale')
    parser.add_argument('--no-quantize', action='store_true', help='Disable INT8 quantization')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert
    onnx_to_tflite(
        args.model,
        args.output,
        args.calib_data,
        args.img_size,
        args.grayscale,
        quantize_int8=not args.no_quantize
    )


if __name__ == '__main__':
    main()
