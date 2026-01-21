"""
Data Generator for Seven-Segment Display ML Training

This module generates clean and noisy training data for machine learning models.
Noise simulation helps the model learn to recognize digits even with sensor errors.
"""

import numpy as np
from typing import Tuple, List
import random


class SevenSegmentDataGenerator:
    """
    Generates training data for seven-segment digit recognition.

    Features:
    - Clean data generation
    - Multiple noise types (flip, missing, extra)
    - Configurable noise levels
    - Balanced dataset generation
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)

        # Ground truth: Perfect seven-segment patterns for digits 0-9
        # Format: (a, b, c, d, e, f, g) segments
        self.digit_patterns = {
            0: [1, 1, 1, 1, 1, 1, 0],
            1: [0, 1, 1, 0, 0, 0, 0],
            2: [1, 1, 0, 1, 1, 0, 1],
            3: [1, 1, 1, 1, 0, 0, 1],
            4: [0, 1, 1, 0, 0, 1, 1],
            5: [1, 0, 1, 1, 0, 1, 1],
            6: [1, 0, 1, 1, 1, 1, 1],
            7: [1, 1, 1, 0, 0, 0, 0],
            8: [1, 1, 1, 1, 1, 1, 1],
            9: [1, 1, 1, 1, 0, 1, 1],
        }

    def generate_clean_data(self, samples_per_digit: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate clean training data without noise.

        Args:
            samples_per_digit: Number of samples to generate per digit (0-9)

        Returns:
            X: Feature array of shape (n_samples, 7)
            y: Label array of shape (n_samples,)
        """
        X = []
        y = []

        for digit in range(10):
            pattern = self.digit_patterns[digit]
            for _ in range(samples_per_digit):
                X.append(pattern)
                y.append(digit)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def add_flip_noise(self, pattern: List[int], noise_level: float = 0.1) -> List[int]:
        """
        Add bit-flip noise: randomly flip 0->1 or 1->0.

        This simulates sensor reading errors or electrical interference.

        Args:
            pattern: Original 7-segment pattern
            noise_level: Probability of flipping each bit (0.0 to 1.0)

        Returns:
            Noisy pattern
        """
        noisy = pattern.copy()
        for i in range(len(noisy)):
            if random.random() < noise_level:
                noisy[i] = 1 - noisy[i]  # Flip bit
        return noisy

    def add_missing_segment_noise(self, pattern: List[int], noise_level: float = 0.1) -> List[int]:
        """
        Add missing segment noise: randomly turn on segments off.

        This simulates burned-out LED segments or weak connections.

        Args:
            pattern: Original 7-segment pattern
            noise_level: Probability of turning off each segment

        Returns:
            Noisy pattern
        """
        noisy = pattern.copy()
        for i in range(len(noisy)):
            if noisy[i] == 1 and random.random() < noise_level:
                noisy[i] = 0  # Turn off segment
        return noisy

    def add_extra_segment_noise(self, pattern: List[int], noise_level: float = 0.1) -> List[int]:
        """
        Add extra segment noise: randomly turn off segments on.

        This simulates ghosting or cross-talk in displays.

        Args:
            pattern: Original 7-segment pattern
            noise_level: Probability of turning on each segment

        Returns:
            Noisy pattern
        """
        noisy = pattern.copy()
        for i in range(len(noisy)):
            if noisy[i] == 0 and random.random() < noise_level:
                noisy[i] = 1  # Turn on segment
        return noisy

    def generate_noisy_data(
        self,
        samples_per_digit: int = 100,
        noise_level: float = 0.15,
        noise_types: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate noisy training data with various noise types.

        Args:
            samples_per_digit: Number of samples per digit
            noise_level: Amount of noise to apply (0.0 to 1.0)
            noise_types: List of noise types to apply. Options:
                        ['flip', 'missing', 'extra', 'mixed']
                        Default: ['flip']

        Returns:
            X: Noisy feature array of shape (n_samples, 7)
            y: Label array of shape (n_samples,)
        """
        if noise_types is None:
            noise_types = ['flip']

        X = []
        y = []

        for digit in range(10):
            pattern = self.digit_patterns[digit]

            for _ in range(samples_per_digit):
                # Choose a random noise type
                noise_type = random.choice(noise_types)

                if noise_type == 'flip':
                    noisy_pattern = self.add_flip_noise(pattern, noise_level)
                elif noise_type == 'missing':
                    noisy_pattern = self.add_missing_segment_noise(pattern, noise_level)
                elif noise_type == 'extra':
                    noisy_pattern = self.add_extra_segment_noise(pattern, noise_level)
                elif noise_type == 'mixed':
                    # Apply multiple noise types
                    noisy_pattern = self.add_flip_noise(pattern, noise_level / 2)
                    noisy_pattern = self.add_missing_segment_noise(noisy_pattern, noise_level / 3)
                    noisy_pattern = self.add_extra_segment_noise(noisy_pattern, noise_level / 3)
                else:
                    noisy_pattern = pattern

                X.append(noisy_pattern)
                y.append(digit)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def generate_mixed_dataset(
        self,
        clean_samples: int = 200,
        noisy_samples: int = 800,
        noise_level: float = 0.15,
        noise_types: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a mixed dataset with both clean and noisy data.

        This is useful for training robust models that can handle both scenarios.

        Args:
            clean_samples: Total clean samples (distributed across 10 digits)
            noisy_samples: Total noisy samples (distributed across 10 digits)
            noise_level: Noise intensity for noisy samples
            noise_types: Types of noise to apply

        Returns:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (clean data)
            y_test: Test labels
        """
        # Generate clean data for testing
        X_clean, y_clean = self.generate_clean_data(samples_per_digit=clean_samples // 10)

        # Generate noisy data for training
        X_noisy, y_noisy = self.generate_noisy_data(
            samples_per_digit=noisy_samples // 10,
            noise_level=noise_level,
            noise_types=noise_types
        )

        # Combine for training set
        X_train = np.vstack([X_clean, X_noisy])
        y_train = np.hstack([y_clean, y_noisy])

        # Generate separate clean test set
        X_test, y_test = self.generate_clean_data(samples_per_digit=50)

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        return X_train, y_train, X_test, y_test

    def visualize_pattern(self, pattern: List[int], label: str = "") -> str:
        """
        Create ASCII visualization of a seven-segment pattern.

        Args:
            pattern: 7-segment pattern [a,b,c,d,e,f,g]
            label: Optional label to display

        Returns:
            ASCII art string representation
        """
        a, b, c, d, e, f, g = pattern

        lines = []
        if label:
            lines.append(f"Digit: {label}")

        # Seven segment layout:
        #  aaa
        # f   b
        # f   b
        #  ggg
        # e   c
        # e   c
        #  ddd

        lines.append(" " + ("_" if a else " ") * 3)
        lines.append(("│" if f else " ") + "   " + ("│" if b else " "))
        lines.append(("│" if f else " ") + "   " + ("│" if b else " "))
        lines.append(" " + ("_" if g else " ") * 3)
        lines.append(("│" if e else " ") + "   " + ("│" if c else " "))
        lines.append(("│" if e else " ") + "   " + ("│" if c else " "))
        lines.append(" " + ("_" if d else " ") * 3)

        return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    generator = SevenSegmentDataGenerator()

    print("=== Clean Data Example ===")
    X_clean, y_clean = generator.generate_clean_data(samples_per_digit=5)
    print(f"Generated {len(X_clean)} clean samples")
    print(f"First sample: {X_clean[0]} -> Label: {y_clean[0]}")
    print(generator.visualize_pattern(X_clean[0].tolist(), str(y_clean[0])))

    print("\n=== Noisy Data Example ===")
    X_noisy, y_noisy = generator.generate_noisy_data(
        samples_per_digit=5,
        noise_level=0.2,
        noise_types=['flip', 'missing', 'extra']
    )
    print(f"Generated {len(X_noisy)} noisy samples")
    print(f"First noisy sample: {X_noisy[0]} -> Label: {y_noisy[0]}")
    print(generator.visualize_pattern(X_noisy[0].tolist(), str(y_noisy[0])))
