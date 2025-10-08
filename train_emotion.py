"""
Train emotion classification model on FER2013 or AffectNet.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent))
from models.emotion_small_cnn import create_emotion_model
from data.datamodule import EmotionDataModule


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{running_loss / len(pbar):.4f}',
                          'acc': f'{100. * correct / total:.2f}%'})

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train emotion classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--img-size', type=int, default=64, help='Input image size')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--output', type=str, default='checkpoints/emotion_small.pt',
                        help='Output model path')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Data
    datamodule = EmotionDataModule(
        args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grayscale=args.grayscale,
        augment=True
    )

    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    # Model
    model = create_emotion_model(num_classes=7, grayscale=args.grayscale)
    model = model.to(device)

    # Loss with class weighting
    class_weights = datamodule.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'args': vars(args)
            }, args.output)
            print(f"✓ Saved best model with accuracy: {best_acc:.2f}%")

    # Save training history
    history_path = output_path.parent / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete! Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
