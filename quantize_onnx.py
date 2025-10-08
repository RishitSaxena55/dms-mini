"""
Quantize ONNX model to INT8 using ONNXRuntime.
"""

import argparse
import numpy as np
from pathlib import Path
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import sys

sys.path.append(str(Path(__file__).parent))
from data.datamodule import CalibrationDataset


class ONNXCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for ONNX quantization."""

    def __init__(self, calib_data_dir, img_size=64, grayscale=False, num_samples=100):
        self.dataset = CalibrationDataset(calib_data_dir, img_size, grayscale, num_samples)
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.dataset)

        try:
            img = next(self.enum_data)
            return {'input': img.numpy()[np.newaxis, ...]}
        except StopIteration:
            return None

    def rewind(self):
        self.enum_data = None


def quantize_onnx_model(model_path, output_path, calib_data_dir, img_size=64, grayscale=False):
    """Quantize ONNX model to INT8."""

    print(f"Quantizing {model_path}...")
    print(f"Using calibration data from {calib_data_dir}")

    # Create calibration reader
    calib_reader = ONNXCalibrationDataReader(calib_data_dir, img_size, grayscale, num_samples=100)

    # Quantize
    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantType.QInt8,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        optimize_model=True
    )

    print(f"✓ Quantized model saved to {output_path}")

    # Compare sizes
    original_size = Path(model_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX model to INT8')
    parser.add_argument('--model', type=str, required=True, help='Input ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Output quantized model')
    parser.add_argument('--calib-data', type=str, required=True,
                        help='Calibration data directory')
    parser.add_argument('--img-size', type=int, default=64, help='Input image size')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Quantize
    quantize_onnx_model(
        args.model,
        args.output,
        args.calib_data,
        args.img_size,
        args.grayscale
    )


if __name__ == '__main__':
    main()
