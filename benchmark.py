"""
Benchmark inference latency and throughput on CPU.
"""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent))
from detect.blazeface_detector import BlazeFaceDetector


def benchmark_onnx(model_path, input_shape, num_frames=300, warmup=50, threads=4):
    """Benchmark ONNX model."""
    import onnxruntime as ort

    # Set threads
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = threads

    session = ort.InferenceSession(model_path, sess_options)

    # Dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        session.run(None, {'input': dummy_input})

    # Benchmark
    latencies = []
    for _ in range(num_frames):
        start = time.perf_counter()
        session.run(None, {'input': dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return latencies


def benchmark_tflite(model_path, input_shape, num_frames=300, warmup=50, threads=4):
    """Benchmark TFLite model."""
    import tensorflow as tf

    # Load model
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=threads
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dummy input
    dummy_input = np.random.randn(*input_shape).astype(input_details[0]['dtype'])

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(num_frames):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return latencies


def benchmark_detector(detector, num_frames=300):
    """Benchmark face detector."""
    # Generate dummy frames
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    latencies = []
    for _ in range(num_frames):
        start = time.perf_counter()
        detector.detect(dummy_frame)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return latencies


def print_stats(latencies, name):
    """Print latency statistics."""
    latencies = np.array(latencies)
    print(f"\n{name}:")
    print(f"  Mean: {latencies.mean():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P90: {np.percentile(latencies, 90):.2f} ms")
    print(f"  P99: {np.percentile(latencies, 99):.2f} ms")
    print(f"  Throughput: {1000 / latencies.mean():.1f} FPS")

    return {
        'mean_ms': float(latencies.mean()),
        'median_ms': float(np.median(latencies)),
        'p90_ms': float(np.percentile(latencies, 90)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'fps': float(1000 / latencies.mean())
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference performance')
    parser.add_argument('--classifier', type=str, required=True,
                        help='Classifier model path (.onnx or .tflite)')
    parser.add_argument('--detector', type=str, default='blazeface',
                        help='Detector type (blazeface or none)')
    parser.add_argument('--img-size', type=int, default=64, help='Input image size')
    parser.add_argument('--channels', type=int, default=1, help='Input channels')
    parser.add_argument('--frames', type=int, default=300, help='Number of frames')
    parser.add_argument('--warmup', type=int, default=50, help='Warmup iterations')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads')
    parser.add_argument('--output', type=str, default='results/benchmark.json',
                        help='Output JSON file')

    args = parser.parse_args()

    print("=" * 60)
    print("Benchmarking Inference Performance")
    print("=" * 60)
    print(f"Classifier: {args.classifier}")
    print(f"Threads: {args.threads}")
    print(f"Frames: {args.frames}")
    print(f"Warmup: {args.warmup}")

    results = {}

    # Benchmark detector
    if args.detector != 'none':
        print("\n[1/3] Benchmarking face detector...")
        detector = BlazeFaceDetector()
        det_latencies = benchmark_detector(detector, args.frames)
        results['detector'] = print_stats(det_latencies, "Face Detector")

    # Benchmark classifier
    input_shape = (1, args.channels, args.img_size, args.img_size)

    if args.classifier.endswith('.onnx'):
        print("\n[2/3] Benchmarking ONNX classifier...")
        cls_latencies = benchmark_onnx(
            args.classifier, input_shape, args.frames, args.warmup, args.threads
        )
    elif args.classifier.endswith('.tflite'):
        print("\n[2/3] Benchmarking TFLite classifier...")
        cls_latencies = benchmark_tflite(
            args.classifier, input_shape, args.frames, args.warmup, args.threads
        )
    else:
        raise ValueError("Classifier must be .onnx or .tflite")

    results['classifier'] = print_stats(cls_latencies, "Emotion Classifier")

    # End-to-end latency
    if args.detector != 'none':
        print("\n[3/3] End-to-end latency...")
        e2e_latencies = np.array(det_latencies) + np.array(cls_latencies)
        results['end_to_end'] = print_stats(e2e_latencies, "End-to-End Pipeline")

    # Model size
    model_path = Path(args.classifier)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    results['model_size_mb'] = model_size_mb

    print(f"\nModel size: {model_size_mb:.2f} MB")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
