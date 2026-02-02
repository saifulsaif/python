# Seven-Segment Digit Recognition 🤖

**A Beginner-Friendly Machine Learning Project**

Learn machine learning by building a neural network that recognizes digits (0-9) from seven-segment displays!

---

## 📖 What is This Project?

This project teaches a computer to recognize digits from seven-segment displays (like those on digital clocks). It's perfect for students learning machine learning concepts!

### Seven-Segment Display Example:
```
 ___      Display for digit "0"
|   |     Segments: a,b,c,d,e,f are ON
|   |     Segment: g is OFF
 ___      Pattern: [1,1,1,1,1,1,0]
```

---

## 🎯 What You'll Learn

- **Neural Networks**: How computers learn patterns
- **Training**: Teaching AI with examples
- **Testing**: Evaluating model accuracy
- **PyTorch**: Popular deep learning framework
- **Data Generation**: Creating training datasets
- **Model Evaluation**: Understanding accuracy and loss

---

## 🚀 Quick Start Guide

### Step 1: Install Requirements

Make sure you have Python 3.7+ installed. Then install the required packages:

```bash
pip3 install -r requirements.txt
```

This installs:
- `torch` - Deep learning framework
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning tools
- `matplotlib` - Visualization (optional)

### Step 2: Run the Program

```bash
python3 main.py
```

You'll see a menu with 6 options. Start with Option 1!

---

## 📋 How to Use (Step-by-Step)

### Option 1: 🎓 TRAIN the Model

**What it does:** Teaches the neural network to recognize digits

**Steps:**
1. Select option `1` from the menu
2. Wait while the model trains (about 1-2 minutes)
3. Watch the accuracy improve with each epoch!
4. The trained model is saved as `best_model.pth`

**Expected Output:**
```
Epoch [10/50]
  Train Loss: 0.1234, Train Acc: 95.50%
  Val Loss: 0.0987, Val Acc: 96.20%
```

**What's happening:**
- The computer generates 1000 training examples
- It learns patterns by adjusting internal parameters
- Accuracy should reach 95%+ by the end

---

### Option 2: 🧪 TEST the Model

**What it does:** Checks how well the trained model performs on new data

**Steps:**
1. Train the model first (Option 1)
2. Select option `2` from the menu
3. The program generates new test examples
4. See how many predictions are correct!

**Expected Output:**
```
📊 Test Results:
   Total examples: 200
   Correct predictions: 192
   Accuracy: 96.00%
```

**What's happening:**
- Creates brand new digit patterns the model hasn't seen
- Tests if the model can generalize (recognize new examples)
- Shows sample predictions with actual vs predicted

---

### Option 3: 🔮 PREDICT a Digit

**What it does:** Let you input your own seven-segment pattern

**Steps:**
1. Train the model first (Option 1)
2. Select option `3` from the menu
3. Enter a pattern when prompted (e.g., `1 1 1 1 1 1 0` for digit 0)
4. See what the model predicts!

**Example Input:**
```
Enter pattern: 1 1 1 1 1 1 0
```

**Expected Output:**
```
📺 Your seven-segment display looks like:
 ___
|   |
|   |
 ___

🎯 Prediction: 0
📊 Confidence: 99.8%
```

**Try these patterns:**
- Digit 0: `1 1 1 1 1 1 0`
- Digit 1: `0 1 1 0 0 0 0`
- Digit 2: `1 1 0 1 1 0 1`
- Digit 8: `1 1 1 1 1 1 1`

---

### Option 4: 📺 SHOW Digit Patterns

**What it does:** Displays all seven-segment patterns for digits 0-9

**Steps:**
1. Select option `4` from the menu
2. Browse through all digit patterns with ASCII visualizations

**Use this to:**
- Understand how seven-segment displays work
- Find patterns to test in Option 3
- Learn the structure of the input data

---

### Option 5: 📚 LEARNING Resources

**What it does:** Explains key machine learning concepts

**Covers:**
- Neural networks explained simply
- Training vs Testing
- Accuracy and Loss metrics
- Links to helpful tutorials
- Project file explanations

---

## 📁 Project Structure

```
seven-segment/
├── main.py                    # Main program (run this!)
├── ml_model.py                # Neural network definition
├── data_generator.py          # Creates training data
├── SevenSegmentDisplay.py     # Digit decoding logic
├── requirements.txt           # Python packages needed
├── README.md                  # This file!
└── best_model.pth            # Saved trained model (created after training)
```

---

## 🧪 Complete Workflow Example

Here's a complete session from start to finish:

### 1. First Time Setup
```bash
# Install packages
pip3 install -r requirements.txt

# Run the program
python3 main.py
```

### 2. Train the Model
```
Choose option: 1

[Training starts...]
Epoch [10/50]
  Train Loss: 0.0856, Train Acc: 97.20%
  Val Loss: 0.0734, Val Acc: 98.00%
...
✓ Training complete!
🎯 Final Test Accuracy: 98.00%
```

### 3. Test the Model
```
Choose option: 2

📊 Test Results:
   Total examples: 200
   Correct predictions: 196
   Accuracy: 98.00%
```

### 4. Try Your Own Predictions
```
Choose option: 3

Enter pattern: 0 1 1 0 0 0 0

🎯 Prediction: 1
📊 Confidence: 99.2%
```

---

## 🎓 Understanding the Code (For Students)

### How Does Training Work?

1. **Generate Data**: Create 1000 examples of digits (0-9)
2. **Feed to Network**: Show examples to the neural network
3. **Calculate Error**: See how wrong the predictions are
4. **Adjust Weights**: Change internal parameters to reduce error
5. **Repeat**: Do this many times (50 epochs)

### What is a Neural Network?

Think of it like a function:
```
Input (7 segments) → [Neural Network] → Output (digit 0-9)
```

The network learns the relationship between input and output by seeing many examples.

### Key Parameters Explained

- **Epochs (50)**: How many times to go through all training data
- **Batch Size (32)**: Process 32 examples at a time
- **Learning Rate (0.001)**: How fast the model learns (not too fast, not too slow)
- **Hidden Layers [32, 16]**: Two layers with 32 and 16 neurons

---

## 🔧 Customization & Experiments

Want to experiment? Try modifying these in `main.py`:

### Change Training Data Amount
```python
# In train_model() function
X_train, y_train, X_test, y_test = generator.generate_mixed_dataset(
    clean_samples=400,    # Try 100, 400, 1000
    noisy_samples=1600,   # Try 400, 1600, 3000
    noise_level=0.15,     # Try 0.0, 0.1, 0.3
)
```

### Change Number of Epochs
```python
# In train_model() function
trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=100,  # Try 30, 100, 200
    verbose=True
)
```

### Change Network Architecture
```python
# In train_model() function
model = SevenSegmentNN(
    hidden_layers=[64, 32, 16],  # Try [16], [64, 32], [128, 64, 32]
    dropout_rate=0.3,            # Try 0.1, 0.5
)
```

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
**Solution:** Install requirements first:
```bash
pip3 install -r requirements.txt
```

### "FileNotFoundError: best_model.pth not found"
**Solution:** Train the model first (Option 1) before testing or predicting

### "Accuracy is low (below 90%)"
**Solution:** Try:
- Training for more epochs (change 50 to 100)
- Using more training data
- Reducing noise level in data generation

### Model takes too long to train
**Solution:** This is normal! Neural network training takes time. On a typical laptop:
- 50 epochs: ~1-2 minutes
- 100 epochs: ~3-5 minutes

---

## 📚 Learning Resources

### Recommended Tutorials

1. **Neural Networks Explained**
   - [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks) - Best visual explanation!
   - [Neural Networks from Scratch](https://www.youtube.com/watch?v=aircAruvnKk)

2. **PyTorch Tutorials**
   - [Official PyTorch Tutorial](https://pytorch.org/tutorials/)
   - [PyTorch in 60 Minutes](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

3. **Machine Learning Basics**
   - [Google's ML Crash Course](https://developers.google.com/machine-learning/crash-course)
   - [Kaggle Learn](https://www.kaggle.com/learn)

---

## 🤝 Contributing

This is a learning project! Feel free to:
- Experiment with the code
- Add new features
- Create visualizations
- Share your modifications

---

## 📝 License

This project is for educational purposes. Feel free to use, modify, and share!

---

## 🎉 Next Steps

After mastering this project, try:

1. **Add more noise types** to make training harder
2. **Visualize training progress** with matplotlib graphs
3. **Try different network architectures** (deeper, wider)
4. **Implement early stopping** to prevent overfitting
5. **Add a web interface** using Flask or Streamlit
6. **Recognize handwritten digits** with the MNIST dataset

---

## 💡 Tips for Success

✅ **Do:**
- Run Option 1 (Train) first
- Experiment with different parameters
- Try to understand each step
- Ask questions and research concepts
- Have fun learning!

❌ **Don't:**
- Skip training and try to test
- Worry if you don't understand everything immediately
- Be afraid to break things (you can always reset)
- Compare yourself to others (everyone learns at their own pace)

---

## 🌟 Congratulations!

You've completed your first machine learning project! You now understand:
- How neural networks learn
- The training and testing process
- How to evaluate model performance
- Basic PyTorch usage

**Keep learning and building!** 🚀

---

**Happy Learning!** If you have questions, check the Learning Resources section or experiment with the code. The best way to learn is by doing!
