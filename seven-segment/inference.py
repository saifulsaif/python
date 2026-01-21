"""
Inference Script - Use Trained Model for Predictions

This script demonstrates how to load a trained model and make predictions
on new seven-segment patterns.
"""

import torch
import numpy as np
from ml_model import SevenSegmentNN
from data_generator import SevenSegmentDataGenerator


class SevenSegmentPredictor:
    """
    Easy-to-use predictor class for seven-segment digit recognition.
    """

    def __init__(self, model_path: str = 'best_model.pth', device: str = 'cpu'):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.generator = SevenSegmentDataGenerator()

        # Initialize model with same architecture as training
        self.model = SevenSegmentNN(
            hidden_layers=[64, 32],
            dropout_rate=0.3
        )

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f"✓ Model loaded from {model_path}")

    def predict_single(self, pattern: list) -> tuple:
        """
        Predict digit from a single seven-segment pattern.

        Args:
            pattern: List of 7 binary values [a,b,c,d,e,f,g]

        Returns:
            (predicted_digit, confidence, all_probabilities)
        """
        # Validate input
        if len(pattern) != 7:
            raise ValueError("Pattern must have exactly 7 segments")

        if not all(bit in [0, 1] for bit in pattern):
            raise ValueError("All segments must be 0 or 1")

        # Convert to tensor
        X = torch.tensor([pattern], dtype=torch.float32).to(self.device)

        # Get predictions
        with torch.no_grad():
            logits = self.model(X)
            probabilities = torch.softmax(logits, dim=1)

        # Extract results
        predicted_digit = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_digit].item() * 100
        all_probs = probabilities[0].cpu().numpy()

        return predicted_digit, confidence, all_probs

    def predict_batch(self, patterns: list) -> tuple:
        """
        Predict digits from multiple patterns at once.

        Args:
            patterns: List of patterns, each with 7 segments

        Returns:
            (predictions, confidences)
        """
        X = torch.tensor(patterns, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X)
            probabilities = torch.softmax(logits, dim=1)

        predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
        confidences = torch.max(probabilities, dim=1)[0].cpu().numpy() * 100

        return predictions, confidences

    def visualize_prediction(self, pattern: list, show_probs: bool = True):
        """
        Make prediction and display visualization.

        Args:
            pattern: Seven-segment pattern
            show_probs: Whether to show probability distribution
        """
        predicted_digit, confidence, all_probs = self.predict_single(pattern)

        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)

        # Show segment visualization
        print("\nSeven-Segment Display:")
        print(self.generator.visualize_pattern(pattern))

        print(f"\nInput Pattern: {pattern}")
        print(f"Predicted Digit: {predicted_digit}")
        print(f"Confidence: {confidence:.1f}%")

        if show_probs:
            print("\nProbability Distribution:")
            for digit, prob in enumerate(all_probs):
                bar_length = int(prob * 50)
                bar = "█" * bar_length
                print(f"  {digit}: {prob*100:5.1f}% {bar}")

        print("=" * 50 + "\n")

        return predicted_digit, confidence

    def test_noise_robustness(self, digit: int, noise_levels: list = None):
        """
        Test model's robustness to noise for a specific digit.

        Args:
            digit: Digit to test (0-9)
            noise_levels: List of noise levels to test
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

        print(f"\n{'='*60}")
        print(f"Noise Robustness Test for Digit {digit}")
        print(f"{'='*60}\n")

        clean_pattern = self.generator.digit_patterns[digit]

        for noise in noise_levels:
            # Generate noisy version
            if noise == 0.0:
                noisy_pattern = clean_pattern
            else:
                noisy_pattern = self.generator.add_flip_noise(clean_pattern, noise)

            # Predict
            pred_digit, confidence, _ = self.predict_single(noisy_pattern)

            # Display result
            status = "✓ CORRECT" if pred_digit == digit else "✗ WRONG"
            print(f"Noise: {noise:.1f} | Pattern: {noisy_pattern} | Pred: {pred_digit} | Conf: {confidence:.1f}% | {status}")


def main():
    """
    Example usage of the inference system.
    """
    print("=" * 60)
    print("Seven-Segment Digit Recognition - Inference Demo")
    print("=" * 60)

    # Initialize predictor
    predictor = SevenSegmentPredictor('best_model.pth')

    # Example 1: Predict clean patterns
    print("\n" + "="*60)
    print("Example 1: Clean Pattern Predictions")
    print("="*60)

    test_patterns = [
        [1, 1, 1, 1, 1, 1, 0],  # 0
        [0, 1, 1, 0, 0, 0, 0],  # 1
        [1, 1, 0, 1, 1, 0, 1],  # 2
        [1, 1, 1, 1, 1, 1, 1],  # 8
    ]

    expected = [0, 1, 2, 8]

    for pattern, expected_digit in zip(test_patterns, expected):
        pred, conf = predictor.visualize_prediction(pattern, show_probs=False)
        if pred == expected_digit:
            print(f"✓ Correct! Expected {expected_digit}, got {pred}\n")
        else:
            print(f"✗ Incorrect! Expected {expected_digit}, got {pred}\n")

    # Example 2: Predict noisy patterns
    print("\n" + "="*60)
    print("Example 2: Noisy Pattern Predictions")
    print("="*60)

    noisy_patterns = [
        [1, 1, 0, 1, 1, 1, 0],  # Noisy 0 (segment c is off)
        [0, 1, 1, 0, 1, 0, 0],  # Noisy 1 (segment e is on - error)
        [1, 1, 1, 1, 1, 1, 0],  # Noisy 8 (segment g is off)
    ]

    print("\nThese patterns have noise - let's see if the model can still recognize them:")

    for pattern in noisy_patterns:
        predictor.visualize_prediction(pattern, show_probs=True)

    # Example 3: Batch prediction
    print("\n" + "="*60)
    print("Example 3: Batch Predictions")
    print("="*60)

    batch_patterns = [
        [1, 1, 1, 1, 1, 1, 0],  # 0
        [0, 1, 1, 0, 0, 0, 0],  # 1
        [1, 1, 0, 1, 1, 0, 1],  # 2
        [1, 1, 1, 1, 0, 0, 1],  # 3
        [0, 1, 1, 0, 0, 1, 1],  # 4
    ]

    predictions, confidences = predictor.predict_batch(batch_patterns)

    print("\nBatch Prediction Results:")
    print("-" * 60)
    for i, (pattern, pred, conf) in enumerate(zip(batch_patterns, predictions, confidences)):
        print(f"Sample {i+1}: {pattern} → Digit {pred} (confidence: {conf:.1f}%)")

    # Example 4: Noise robustness test
    print("\n" + "="*60)
    print("Example 4: Testing Noise Robustness")
    print("="*60)

    predictor.test_noise_robustness(digit=8, noise_levels=[0.0, 0.1, 0.15, 0.2, 0.25, 0.3])

    # Example 5: Interactive mode
    print("\n" + "="*60)
    print("Example 5: Custom Input")
    print("="*60)

    print("\nTry entering your own pattern!")
    print("Enter 7 binary values (0 or 1) separated by spaces")
    print("Example: 1 1 1 1 1 1 0")
    print("\n(Press Ctrl+C to skip)")

    try:
        user_input = input("\nEnter pattern: ")
        pattern = [int(x) for x in user_input.split()]

        if len(pattern) == 7 and all(x in [0, 1] for x in pattern):
            predictor.visualize_prediction(pattern, show_probs=True)
        else:
            print("Invalid input! Please enter exactly 7 binary values (0 or 1).")

    except (KeyboardInterrupt, EOFError):
        print("\nSkipping interactive mode.")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*60)
    print("Inference Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    # Check if model exists
    import os
    if not os.path.exists('best_model.pth'):
        print("❌ Error: Model file 'best_model.pth' not found!")
        print("\nPlease train the model first by running:")
        print("  python train_model.py")
        print("\nThis will create the 'best_model.pth' file.")
    else:
        main()
