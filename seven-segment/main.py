"""
SEVEN-SEGMENT DIGIT RECOGNITION - MAIN PROGRAM
===============================================

This is the main control program for learning machine learning!

What this project does:
- Trains a neural network to recognize digits (0-9) from seven-segment displays
- Tests how well the model learned
- Makes predictions on new data

Perfect for beginners learning machine learning concepts!
"""

import torch
import numpy as np
from data_generator import SevenSegmentDataGenerator
from ml_model import SevenSegmentNN, SevenSegmentTrainer
from SevenSegmentDisplay import SevenSegmentDisplay
import sys


def print_header(text):
    """Print a nice header for sections."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def train_model():
    """
    STEP 1: Train the Neural Network

    This teaches the computer to recognize digits by showing it many examples.
    Like teaching a child by showing flashcards!
    """
    print_header("TRAINING THE MODEL")

    print("\n📚 What is happening:")
    print("   - Creating fake seven-segment patterns (training data)")
    print("   - Teaching the neural network to recognize each digit")
    print("   - The network adjusts itself to get better with each example")

    # Step 1: Generate training data
    print("\n1️⃣  Generating training data...")
    generator = SevenSegmentDataGenerator(seed=42)

    # Create a dataset with clean and noisy examples
    # This makes the model more robust (better at handling errors)
    X_train, y_train, X_test, y_test = generator.generate_mixed_dataset(
        clean_samples=200,    # 200 perfect examples
        noisy_samples=800,    # 800 examples with some errors
        noise_level=0.15,     # 15% chance of error in each segment
        noise_types=['flip', 'missing', 'extra']
    )

    print(f"   ✓ Created {len(X_train)} training examples")
    print(f"   ✓ Created {len(X_test)} test examples")
    print(f"   ✓ Each example has 7 features (one per segment)")

    # Step 2: Convert data to PyTorch format
    print("\n2️⃣  Preparing data for PyTorch...")
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

    # DataLoader: Feeds data to the model in small batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,    # Process 32 examples at a time
        shuffle=True      # Randomize order for better learning
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )
    print("   ✓ Data is ready!")

    # Step 3: Create the neural network
    print("\n3️⃣  Creating the neural network...")
    print("   Architecture: 7 inputs -> 32 neurons -> 16 neurons -> 10 outputs")
    print("   (7 segments in, 10 possible digits out)")

    model = SevenSegmentNN(
        hidden_layers=[32, 16],  # Two hidden layers
        dropout_rate=0.3,        # Prevents overfitting
        use_batch_norm=True      # Makes training more stable
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {total_params} learnable parameters")

    # Step 4: Train the model
    print("\n4️⃣  Training the model...")
    print("   This will take a minute. Watch the accuracy improve!\n")

    trainer = SevenSegmentTrainer(
        model=model,
        learning_rate=0.001,  # How fast the model learns
        device='cpu'           # Use CPU (change to 'cuda' if you have GPU)
    )

    # Train for 50 epochs (one epoch = seeing all training data once)
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=50,
        verbose=True
    )

    print("\n   ✓ Training complete!")
    print(f"   ✓ Best model saved to: best_model.pth")

    # Step 5: Final evaluation
    print_header("FINAL EVALUATION")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")
    print(f"📊 Final Test Loss: {test_loss:.4f}")

    if test_acc >= 95:
        print("\n🌟 Excellent! Your model is highly accurate!")
    elif test_acc >= 85:
        print("\n👍 Good! Your model learned well!")
    else:
        print("\n💡 Tip: Try training longer or adjusting parameters.")

    return model, trainer


def test_model():
    """
    STEP 2: Test the Trained Model

    Load the saved model and test it on new examples.
    """
    print_header("TESTING THE MODEL")

    try:
        # Load the trained model
        print("\n📂 Loading saved model...")
        model = SevenSegmentNN(hidden_layers=[32, 16])
        trainer = SevenSegmentTrainer(model=model)
        trainer.load_model('best_model.pth')
        print("   ✓ Model loaded successfully!")

        # Generate test data
        print("\n🧪 Generating test examples...")
        generator = SevenSegmentDataGenerator(seed=99)  # Different seed for new data
        X_test, y_test = generator.generate_noisy_data(
            samples_per_digit=20,
            noise_level=0.1,
            noise_types=['flip']
        )

        # Make predictions
        print("\n🔮 Making predictions...")
        X_tensor = torch.tensor(X_test)
        predictions = model.predict(X_tensor)

        # Calculate accuracy
        correct = (predictions.numpy() == y_test).sum()
        accuracy = 100 * correct / len(y_test)

        print(f"\n📊 Test Results:")
        print(f"   Total examples: {len(y_test)}")
        print(f"   Correct predictions: {correct}")
        print(f"   Accuracy: {accuracy:.2f}%")

        # Show some examples
        print("\n📋 Sample Predictions (first 10):")
        print("   Actual | Predicted | Correct?")
        print("   " + "-" * 35)
        for i in range(min(10, len(y_test))):
            actual = y_test[i]
            pred = predictions[i].item()
            correct_mark = "✓" if actual == pred else "✗"
            print(f"     {actual}    |     {pred}     |    {correct_mark}")

        return model

    except FileNotFoundError:
        print("\n❌ Error: Model file 'best_model.pth' not found!")
        print("   Please train the model first using option 1.")
        return None


def predict_single():
    """
    STEP 3: Make a Single Prediction

    Input a seven-segment pattern and see what digit the model predicts.
    """
    print_header("MAKE A PREDICTION")

    try:
        # Load model
        print("\n📂 Loading model...")
        model = SevenSegmentNN(hidden_layers=[32, 16])
        trainer = SevenSegmentTrainer(model=model)
        trainer.load_model('best_model.pth')
        print("   ✓ Model loaded!")

        # Get user input
        print("\n💡 Enter a seven-segment pattern (7 bits: 0 or 1)")
        print("   Segments: a b c d e f g")
        print("   Example: 1 1 1 1 1 1 0  (represents digit 0)")
        print("\n   Common patterns:")
        print("   0: 1 1 1 1 1 1 0")
        print("   1: 0 1 1 0 0 0 0")
        print("   2: 1 1 0 1 1 0 1")
        print("   8: 1 1 1 1 1 1 1")

        user_input = input("\n👉 Enter pattern: ").strip()

        # Parse input
        try:
            segments = [int(x) for x in user_input.split()]
            if len(segments) != 7:
                print("❌ Error: Please enter exactly 7 numbers!")
                return
            if not all(s in [0, 1] for s in segments):
                print("❌ Error: Each segment must be 0 or 1!")
                return
        except ValueError:
            print("❌ Error: Please enter valid numbers!")
            return

        # Visualize the pattern
        print("\n📺 Your seven-segment display looks like:")
        generator = SevenSegmentDataGenerator()
        print(generator.visualize_pattern(segments))

        # Make prediction
        X = torch.tensor([segments], dtype=torch.float32)
        prediction = model.predict(X)
        probabilities = model.predict_proba(X)

        pred_digit = prediction[0].item()
        confidence = probabilities[0][pred_digit].item() * 100

        print(f"\n🎯 Prediction: {pred_digit}")
        print(f"📊 Confidence: {confidence:.1f}%")

        # Show all probabilities
        print("\n📈 Probability for each digit:")
        for digit in range(10):
            prob = probabilities[0][digit].item() * 100
            bar = "█" * int(prob / 5)  # Visual bar
            print(f"   {digit}: {prob:5.1f}% {bar}")

    except FileNotFoundError:
        print("\n❌ Error: Model file not found! Please train the model first.")


def show_digit_patterns():
    """
    STEP 4: Show All Digit Patterns

    Display the seven-segment patterns for all digits 0-9.
    """
    print_header("SEVEN-SEGMENT DIGIT PATTERNS")

    generator = SevenSegmentDataGenerator()

    print("\nHere are the patterns for all digits:\n")

    for digit in range(10):
        pattern = generator.digit_patterns[digit]
        print(f"\n{'=' * 30}")
        print(f"DIGIT: {digit}")
        print(f"Pattern: {pattern}")
        print(generator.visualize_pattern(pattern))


def learning_resources():
    """
    STEP 5: Learning Resources

    Helpful information for understanding the code.
    """
    print_header("LEARNING RESOURCES")

    print("""
📚 KEY CONCEPTS FOR BEGINNERS:

1. NEURAL NETWORK
   - Like a brain made of math
   - Learns patterns from examples
   - Gets better with practice (training)

2. TRAINING
   - Showing the network many examples
   - Network adjusts itself to improve
   - Like practicing math problems to get better

3. TESTING
   - Checking if the network learned correctly
   - Using NEW examples it hasn't seen before
   - Like taking a quiz after studying

4. ACCURACY
   - Percentage of correct predictions
   - 95% accuracy = correct 95 times out of 100
   - Higher is better!

5. EPOCHS
   - One complete pass through all training data
   - More epochs = more practice
   - But too many can cause "overfitting"

6. LOSS
   - How wrong the predictions are
   - Lower loss = better predictions
   - Goal: minimize loss during training

📁 PROJECT FILES:

   main.py              - This file! Main menu/control
   ml_model.py          - Neural network code (PyTorch)
   data_generator.py    - Creates training data
   SevenSegmentDisplay.py - Digit decoding logic
   requirements.txt     - Libraries needed

🔗 USEFUL LINKS:

   PyTorch Tutorial: https://pytorch.org/tutorials/
   Neural Networks: https://www.3blue1brown.com/topics/neural-networks
   Machine Learning Basics: https://developers.google.com/machine-learning

💡 TIPS:

   - Start by training the model (Option 1)
   - Then test it (Option 2) to see how well it learned
   - Try making predictions (Option 3) with different patterns
   - Experiment with the code - that's how you learn!
   - Don't worry if you don't understand everything at first
""")


def main_menu():
    """
    Main Menu - The control center for your ML project!
    """
    while True:
        print("\n" + "=" * 60)
        print("  🤖 SEVEN-SEGMENT DIGIT RECOGNITION")
        print("  Machine Learning for Beginners")
        print("=" * 60)

        print("""
Choose an option:

1. 🎓 TRAIN the model (Teach the computer)
2. 🧪 TEST the model (Check how well it learned)
3. 🔮 PREDICT a digit (Try it yourself)
4. 📺 SHOW digit patterns (See all digits)
5. 📚 LEARNING resources (Understand the code)
6. 🚪 EXIT

What would you like to do?
""")

        choice = input("👉 Enter your choice (1-6): ").strip()

        if choice == '1':
            train_model()
        elif choice == '2':
            test_model()
        elif choice == '3':
            predict_single()
        elif choice == '4':
            show_digit_patterns()
        elif choice == '5':
            learning_resources()
        elif choice == '6':
            print("\n👋 Goodbye! Keep learning!\n")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice! Please enter 1-6.")


if __name__ == '__main__':
    # Start the program
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║     WELCOME TO SEVEN-SEGMENT ML PROJECT!            ║
    ║                                                      ║
    ║     Learn machine learning by building a            ║
    ║     digit recognition system!                       ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    main_menu()
