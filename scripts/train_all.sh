# scripts/train_all.sh
#!/bin/bash
set -e

echo "DMS Mini - Training Pipeline"
echo "=============================="

# Train emotion classifier
echo "[1/2] Training emotion classifier..."
python src/train_emotion.py \
    --data datasets/fer2013 \
    --img-size 64 \
    --grayscale \
    --epochs 40 \
    --batch-size 128 \
    --output checkpoints/emotion_small.pt

# Train seat-belt classifier (bonus)
echo "[2/2] Training seat-belt classifier..."
if [ -d "datasets/seatbelt" ]; then
    python src/train_seatbelt.py \
        --data datasets/seatbelt \
        --img-size 64 \
        --epochs 20 \
        --output checkpoints/seatbelt_tiny.pt
else
    echo "Skipping seat-belt training (dataset not found)"
fi

echo "✓ Training complete!"
