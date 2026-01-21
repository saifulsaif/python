"""
BEGINNER-FRIENDLY Neural Network for Seven-Segment Digit Recognition

What does this do?
------------------
This program teaches a computer to recognize digits (0-9) from seven-segment displays
(like those on digital clocks). The computer learns patterns by looking at examples.

Key Concepts:
- Neural Network: A computer program that learns patterns (like your brain!)
- Training: Teaching the network by showing it many examples
- Testing: Checking if the network learned correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SevenSegmentNN(nn.Module):
    """
    Feedforward Neural Network for Seven-Segment Digit Classification.

    Architecture:
    - Input layer: 7 features (one per segment)
    - Hidden layers: Configurable depth and width
    - Output layer: 10 classes (digits 0-9)

    The model uses:
    - ReLU activation for non-linearity
    - Dropout for regularization (prevents overfitting)
    - Batch normalization for stable training
    """

    def __init__(
        self,
        hidden_layers: list = [32, 16],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize the neural network.

        Args:
            hidden_layers: List of hidden layer sizes. E.g., [32, 16] creates
                          two hidden layers with 32 and 16 neurons.
            dropout_rate: Dropout probability (0.0 to 1.0). Higher values
                         provide more regularization but may hurt performance.
            use_batch_norm: Whether to use batch normalization.
        """
        super(SevenSegmentNN, self).__init__()

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build the network layers dynamically
        layers = []
        input_size = 7  # Seven segments

        # Create hidden layers
        for hidden_size in hidden_layers:
            # Linear transformation
            layers.append(nn.Linear(input_size, hidden_size))

            # Batch normalization (helps with training stability)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # ReLU activation (introduces non-linearity)
            layers.append(nn.ReLU())

            # Dropout (prevents overfitting)
            layers.append(nn.Dropout(dropout_rate))

            input_size = hidden_size

        # Store as sequential model
        self.hidden = nn.Sequential(*layers)

        # Output layer: 10 classes for digits 0-9
        self.output = nn.Linear(input_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 7)

        Returns:
            Output tensor of shape (batch_size, 10) with class logits
        """
        x = self.hidden(x)
        x = self.output(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the model.

        Args:
            x: Input tensor of shape (batch_size, 7)

        Returns:
            Predicted class labels (0-9)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities for each class.

        Args:
            x: Input tensor of shape (batch_size, 7)

        Returns:
            Probability tensor of shape (batch_size, 10)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


class SevenSegmentTrainer:
    """
    Trainer class for the Seven-Segment Neural Network.

    Handles:
    - Model training with backpropagation
    - Validation during training
    - Learning rate scheduling
    - Model checkpointing
    """

    def __init__(
        self,
        model: SevenSegmentNN,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize the trainer.

        Args:
            model: The neural network model to train
            learning_rate: Learning rate for optimization
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # Loss function: Cross-entropy for multi-class classification
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: Adam (adaptive learning rate)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler: Reduces LR when validation plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average loss and accuracy on validation set
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        verbose: bool = True
    ):
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


# Alternative: Simple Sklearn Model for comparison
class SevenSegmentMLPClassifier:
    """
    Simple Multi-Layer Perceptron using scikit-learn.

    This is a simpler alternative to PyTorch for quick experimentation.
    Good for understanding ML basics without deep learning complexity.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (32, 16),
        max_iter: int = 500,
        random_state: int = 42
    ):
        """
        Initialize the MLP classifier.

        Args:
            hidden_layer_sizes: Sizes of hidden layers
            max_iter: Maximum training iterations
            random_state: Random seed for reproducibility
        """
        from sklearn.neural_network import MLPClassifier

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=True
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Get accuracy score."""
        return self.model.score(X, y)


if __name__ == "__main__":
    # Demo: Create and inspect model architecture
    model = SevenSegmentNN(hidden_layers=[32, 16], dropout_rate=0.3)
    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    dummy_input = torch.randn(4, 7)  # Batch of 4 samples
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
