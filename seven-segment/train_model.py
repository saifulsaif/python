"""
Training Script for Seven-Segment Digit Recognition

This script trains a neural network to recognize digits from seven-segment
displays, handling both clean and noisy data.
"""

import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os

from data_generator import SevenSegmentDataGenerator
from ml_model import SevenSegmentNN, SevenSegmentTrainer


def prepare_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Prepare PyTorch DataLoaders from numpy arrays.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size for training

    Returns:
        train_loader, val_loader
    """
    # Convert to PyTorch tensors
    train_dataset = data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    val_dataset = data.TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )

    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def evaluate_model_on_noise_levels(
    model: SevenSegmentNN,
    device: str = 'cpu'
):
    """
    Evaluate model performance across different noise levels.

    This helps understand how robust the model is to noise.

    Args:
        model: Trained model
        device: Device to run evaluation on
    """
    generator = SevenSegmentDataGenerator()
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accuracies = []

    model.eval()
    model.to(device)

    print("\n=== Noise Robustness Evaluation ===")
    print("Noise Level | Accuracy")
    print("-" * 30)

    for noise_level in noise_levels:
        # Generate test data at this noise level
        X_test, y_test = generator.generate_noisy_data(
            samples_per_digit=100,
            noise_level=noise_level,
            noise_types=['flip', 'missing', 'extra', 'mixed']
        )

        # Convert to tensors
        X_tensor = torch.from_numpy(X_test).to(device)
        y_tensor = torch.from_numpy(y_test).to(device)

        # Predict
        predictions = model.predict(X_tensor)

        # Calculate accuracy
        accuracy = (predictions == y_tensor).float().mean().item() * 100
        accuracies.append(accuracy)

        print(f"   {noise_level:.2f}     |  {accuracy:.2f}%")

    # Plot noise robustness
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness to Noise')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=90, color='r', linestyle='--', label='90% threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('noise_robustness.png', dpi=300)
    print("\nNoise robustness plot saved to noise_robustness.png")
    plt.show()


def visualize_predictions(
    model: SevenSegmentNN,
    generator: SevenSegmentDataGenerator,
    num_samples: int = 10,
    noise_level: float = 0.2,
    device: str = 'cpu'
):
    """
    Visualize model predictions on noisy samples.

    Args:
        model: Trained model
        generator: Data generator
        num_samples: Number of samples to visualize
        noise_level: Noise level for test samples
        device: Device to run inference on
    """
    # Generate noisy test samples
    X_test, y_test = generator.generate_noisy_data(
        samples_per_digit=num_samples // 10,
        noise_level=noise_level,
        noise_types=['flip', 'missing', 'extra', 'mixed']
    )

    # Get predictions
    model.eval()
    model.to(device)
    X_tensor = torch.from_numpy(X_test).to(device)
    predictions = model.predict(X_tensor).cpu().numpy()
    probabilities = model.predict_proba(X_tensor).cpu().numpy()

    print("\n=== Sample Predictions ===")
    for i in range(min(num_samples, len(X_test))):
        pattern = X_test[i].astype(int).tolist()
        true_label = y_test[i]
        pred_label = predictions[i]
        confidence = probabilities[i][pred_label] * 100

        print(f"\nSample {i+1}:")
        print(f"  Pattern: {pattern}")
        print(f"  True Label: {true_label}")
        print(f"  Predicted: {pred_label} (confidence: {confidence:.1f}%)")

        if pred_label != true_label:
            print(f"  ❌ INCORRECT")
        else:
            print(f"  ✓ CORRECT")

        # Show visualization
        print(generator.visualize_pattern(pattern))


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Seven-Segment Digit Recognition - Training Pipeline")
    print("=" * 60)

    # Configuration
    config = {
        'clean_samples': 200,
        'noisy_samples': 2000,
        'noise_level': 0.15,
        'noise_types': ['flip', 'missing', 'extra', 'mixed'],
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'hidden_layers': [64, 32],
        'dropout_rate': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Step 1: Generate data
    print("\n[Step 1/5] Generating training data...")
    generator = SevenSegmentDataGenerator(seed=42)

    X_train, y_train, X_val, y_val = generator.generate_mixed_dataset(
        clean_samples=config['clean_samples'],
        noisy_samples=config['noisy_samples'],
        noise_level=config['noise_level'],
        noise_types=config['noise_types']
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Step 2: Prepare data loaders
    print("\n[Step 2/5] Preparing data loaders...")
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=config['batch_size']
    )

    # Step 3: Create model
    print("\n[Step 3/5] Creating neural network model...")
    model = SevenSegmentNN(
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout_rate']
    )
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Step 4: Train model
    print("\n[Step 4/5] Training model...")
    trainer = SevenSegmentTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        device=config['device']
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        verbose=True
    )

    print("\n✓ Training complete!")

    # Step 5: Evaluate and visualize
    print("\n[Step 5/5] Evaluating model...")

    # Plot training history
    plot_training_history(trainer.history, save_path='training_history.png')

    # Evaluate on different noise levels
    evaluate_model_on_noise_levels(model, device=config['device'])

    # Show sample predictions
    visualize_predictions(
        model=model,
        generator=generator,
        num_samples=10,
        noise_level=0.2,
        device=config['device']
    )

    # Final test accuracy
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"\n{'='*60}")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"{'='*60}")

    print("\n✓ Training pipeline complete!")
    print("\nModel saved to: best_model.pth")
    print("Plots saved to: training_history.png, noise_robustness.png")


if __name__ == "__main__":
    main()
