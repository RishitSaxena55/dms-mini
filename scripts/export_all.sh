# scripts/export_all.sh
#!/bin/bash
set -e

echo "DMS Mini - Export and Quantization Pipeline"
echo "============================================"

mkdir -p models results

# Export to ONNX
echo "[1/4] Exporting to ONNX..."
python src/export_onnx.py \
    --weights checkpoints/emotion_small.pt \
    --output models/emotion_small.onnx \
    --img-size 64 \
    --channels 1

# Quantize ONNX
echo "[2/4] Quantizing ONNX to INT8..."
python src/quantize_onnx.py \
    --model models/emotion_small.onnx \
    --calib-data datasets/fer2013/train \
    --output models/emotion_small_int8.onnx \
    --grayscale

# Quantize TFLite
echo "[3/4] Converting to TFLite INT8..."
python src/quantize_tflite.py \
    --model models/emotion_small.onnx \
    --calib-data datasets/fer2013/train \
    --output models/emotion_small_int8.tflite \
    --grayscale

# Benchmark
echo "[4/4] Benchmarking performance..."
python src/benchmark.py \
    --classifier models/emotion_small_int8.tflite \
    --detector blazeface \
    --frames 300 \
    --threads 4 \
    --output results/benchmark.json

echo "✓ Export and quantization complete!"
