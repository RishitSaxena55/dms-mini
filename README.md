# DMS Mini - Driver Monitoring System (Track B)

A lightweight emotion recognition system for driver monitoring using BlazeFace detection and a small CNN classifier trained on FER2013/AffectNet.

## Features

- **Fast Face Detection**: BlazeFace TFLite for real-time CPU inference
- **Small Emotion Classifier**: Custom lightweight CNN (~0.3M parameters)
- **Optimized Models**: ONNX and TFLite with INT8 quantization
- **Bonus**: Seat-belt ROI detection with tiny binary CNN
- **Complete Metrics**: Accuracy, confusion matrix, latency benchmarks, model sizes

## Project Structure

```
dms-mini/
├── README.md
├── requirements.txt
├── environment.yml
├── brief.md                    # Metrics report
├── datasets/
│   ├── download_fer2013.py     # Dataset download script
│   └── README.md
├── src/
│   ├── detect/
│   │   └── blazeface_detector.py
│   ├── data/
│   │   └── datamodule.py
│   ├── models/
│   │   ├── emotion_small_cnn.py
│   │   └── seatbelt_tiny_cnn.py
│   ├── train_emotion.py
│   ├── train_seatbelt.py
│   ├── export_onnx.py
│   ├── quantize_onnx.py
│   ├── quantize_tflite.py
│   ├── benchmark.py
│   ├── eval_metrics.py
│   └── infer_realtime.py
└── scripts/
    ├── train_all.sh
    └── export_all.sh
```

## Setup

### Option 1: pip (recommended)
```bash
pip install -r requirements.txt
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate dms-mini
```

## Quick Start

### 1. Download Dataset

```bash
# Download FER2013 from Kaggle
python datasets/download_fer2013.py --output datasets/fer2013

# Or manually download from:
# https://www.kaggle.com/datasets/msambare/fer2013
# Expected structure: datasets/fer2013/{train,test}/{angry,disgust,fear,happy,sad,surprise,neutral}/
```

### 2. Train Emotion Classifier

```bash
python src/train_emotion.py \
    --data datasets/fer2013 \
    --img-size 64 \
    --epochs 40 \
    --batch-size 128 \
    --output checkpoints/emotion_small.pt
```

### 3. Evaluate Model

```bash
python src/eval_metrics.py \
    --data datasets/fer2013 \
    --weights checkpoints/emotion_small.pt \
    --output results/
```

### 4. Export to ONNX

```bash
python src/export_onnx.py \
    --weights checkpoints/emotion_small.pt \
    --output models/emotion_small.onnx \
    --img-size 64
```

### 5. Quantize to INT8

```bash
# ONNX INT8
python src/quantize_onnx.py \
    --model models/emotion_small.onnx \
    --calib-data datasets/fer2013/train \
    --output models/emotion_small_int8.onnx

# TFLite INT8
python src/quantize_tflite.py \
    --model models/emotion_small.onnx \
    --calib-data datasets/fer2013/train \
    --output models/emotion_small_int8.tflite
```

### 6. Benchmark Performance

```bash
python src/benchmark.py \
    --classifier models/emotion_small_int8.tflite \
    --detector blazeface \
    --frames 300 \
    --threads 4
```

### 7. Real-time Inference

```bash
python src/infer_realtime.py \
    --detector blazeface \
    --classifier models/emotion_small_int8.tflite \
    --camera 0 \
    --seatbelt
```

## Bonus: Seat-belt Detection

Train the tiny seat-belt classifier:

```bash
python src/train_seatbelt.py \
    --data datasets/seatbelt \
    --output checkpoints/seatbelt_tiny.pt
```

## Expected Results

### Accuracy
- **Top-1 Accuracy**: ~65-70% on FER2013 test set
- **Model Size**: 
  - FP32: ~1.2 MB
  - INT8: ~0.3 MB (4× reduction)

### Latency (Intel i5 CPU, 4 threads)
- **BlazeFace Detection**: ~15 ms/frame
- **Emotion Classifier (INT8)**: ~8 ms/frame
- **End-to-end**: ~25 ms/frame (~40 FPS)

## License

MIT License

## Citation

```bibtex
@misc{dms-mini-2025,
  title={DMS Mini: Lightweight Driver Monitoring System},
  author={Your Name},
  year={2025}
}
```
