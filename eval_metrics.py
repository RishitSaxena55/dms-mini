"""
Evaluate model and generate metrics: accuracy, confusion matrix, per-class metrics.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

sys.path.append(str(Path(__file__).parent))
from models.emotion_small_cnn import create_emotion_model
from data.datamodule import EmotionDataModule


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ Confusion matrix saved to {output_path}")


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and collect predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   output_dict=True)

    return accuracy, cm, report


def main():
    parser = argparse.ArgumentParser(description='Evaluate emotion classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--weights', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(args.weights, map_location=device)
    model_args = checkpoint.get('args', {})

    img_size = model_args.get('img_size', 64)
    grayscale = model_args.get('grayscale', False)

    # Data
    datamodule = EmotionDataModule(
        args.data,
        img_size=img_size,
        batch_size=args.batch_size,
        grayscale=grayscale,
        augment=False
    )
    test_loader = datamodule.test_dataloader()
    class_names = datamodule.EMOTIONS

    # Model
    model = create_emotion_model(num_classes=7, grayscale=grayscale)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    print("Evaluating model...")
    accuracy, cm, report = evaluate_model(model, test_loader, device, class_names)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 60}\n")

    print("Per-class metrics:")
    for class_name in class_names:
        metrics = report[class_name]
        print(f"  {class_name:10s}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

    print(f"\nMacro Average: Precision={report['macro avg']['precision']:.3f}, "
          f"Recall={report['macro avg']['recall']:.3f}, "
          f"F1={report['macro avg']['f1-score']:.3f}")

    # Save confusion matrix
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')

    # Save metrics
    metrics_dict = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'model_params': model.count_parameters()
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
