"""
Simple Example using scikit-learn (No PyTorch needed!)

This is a simplified version for beginners who want to understand
machine learning without the complexity of PyTorch.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from data_generator import SevenSegmentDataGenerator


def main():
    """
    Simple ML pipeline using scikit-learn.
    """
    print("=" * 70)
    print("Simple Seven-Segment Recognition with scikit-learn")
    print("=" * 70)

    # Step 1: Generate data
    print("\n[Step 1] Generating training data...")
    generator = SevenSegmentDataGenerator(seed=42)

    # Generate mixed dataset
    X_train, y_train, X_test, y_test = generator.generate_mixed_dataset(
        clean_samples=200,
        noisy_samples=800,
        noise_level=0.15,
        noise_types=['flip', 'missing', 'extra']
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Step 2: Create and train model
    print("\n[Step 2] Training neural network...")

    # Create MLP (Multi-Layer Perceptron) classifier
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Two hidden layers
        activation='relu',             # ReLU activation
        solver='adam',                 # Adam optimizer
        batch_size=32,                 # Batch size
        learning_rate_init=0.001,      # Learning rate
        max_iter=100,                  # Maximum epochs
        random_state=42,               # For reproducibility
        early_stopping=True,           # Stop when validation plateaus
        validation_fraction=0.1,       # Use 10% for validation
        verbose=True                   # Show training progress
    )

    # Train the model
    model.fit(X_train, y_train)

    print("\n✓ Training complete!")

    # Step 3: Evaluate model
    print("\n[Step 3] Evaluating model...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate accuracy
    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100

    print(f"\n  Training Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")

    # Step 4: Detailed classification report
    print("\n[Step 4] Classification Report:")
    print("\n" + classification_report(y_test, y_test_pred))

    # Step 5: Confusion matrix
    print("\n[Step 5] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nRows = True Labels, Columns = Predictions")
    print("    ", "  ".join([str(i) for i in range(10)]))
    for i, row in enumerate(cm):
        print(f"{i}: ", "  ".join([f"{x:2d}" for x in row]))

    # Step 6: Visualize learning curve
    print("\n[Step 6] Plotting learning curve...")

    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, 'b-', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Learning Curve (Training Loss)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sklearn_learning_curve.png', dpi=300)
    print("  Saved to: sklearn_learning_curve.png")
    plt.show()

    # Step 7: Test with noisy data
    print("\n[Step 7] Testing robustness to noise...")

    noise_levels = [0.0, 0.1, 0.2, 0.3]
    accuracies = []

    print("\nNoise Level | Accuracy")
    print("-" * 30)

    for noise in noise_levels:
        X_noisy, y_noisy = generator.generate_noisy_data(
            samples_per_digit=100,
            noise_level=noise,
            noise_types=['flip', 'missing', 'extra']
        )

        y_pred = model.predict(X_noisy)
        acc = accuracy_score(y_noisy, y_pred) * 100
        accuracies.append(acc)

        print(f"   {noise:.1f}      |  {acc:.2f}%")

    # Plot noise robustness
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, accuracies, 'ro-', linewidth=2, markersize=10)
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness to Noise')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig('sklearn_noise_robustness.png', dpi=300)
    print("\n  Saved to: sklearn_noise_robustness.png")
    plt.show()

    # Step 8: Example predictions
    print("\n[Step 8] Example Predictions:")

    test_patterns = [
        ([1, 1, 1, 1, 1, 1, 0], 0),  # Clean 0
        ([0, 1, 1, 0, 0, 0, 0], 1),  # Clean 1
        ([1, 1, 0, 1, 1, 1, 0], 0),  # Noisy 0
        ([1, 1, 1, 1, 1, 1, 1], 8),  # Clean 8
    ]

    print("\nPattern                 | Expected | Predicted | Confidence")
    print("-" * 70)

    for pattern, expected in test_patterns:
        pred = model.predict([pattern])[0]
        proba = model.predict_proba([pattern])[0]
        confidence = proba[pred] * 100

        status = "✓" if pred == expected else "✗"
        print(f"{pattern} | {expected}        | {pred}         | {confidence:.1f}%  {status}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

    # Save the model
    import joblib
    joblib.dump(model, 'sklearn_model.pkl')
    print("\n✓ Model saved to: sklearn_model.pkl")
    print("\nTo load the model later:")
    print("  import joblib")
    print("  model = joblib.load('sklearn_model.pkl')")


if __name__ == "__main__":
    main()
