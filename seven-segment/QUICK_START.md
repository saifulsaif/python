# 🚀 QUICK START GUIDE

**Get started in 2 minutes!**

---

## Step 1: Install (First time only)

```bash
pip3 install -r requirements.txt
```

Wait for PyTorch and other packages to install (this may take a few minutes).

---

## Step 2: Run the Program

```bash
python3 main.py
```

---

## Step 3: Follow the Menu

### First Time? Do This:

1. **Choose Option 1** - Train the model (takes 1-2 minutes)
2. **Choose Option 2** - Test the model
3. **Choose Option 3** - Try your own predictions!

---

## 📋 Quick Commands Reference

### Train the Model
```
1. Run: python3 main.py
2. Choose: 1
3. Wait: ~1-2 minutes
4. Done: Model saved as best_model.pth
```

### Test the Model
```
1. Run: python3 main.py
2. Choose: 2
3. See: Accuracy results
```

### Make a Prediction
```
1. Run: python3 main.py
2. Choose: 3
3. Enter: 1 1 1 1 1 1 0 (for digit 0)
4. See: Prediction result
```

---

## 🔢 Sample Patterns to Try

Copy and paste these into Option 3:

- **Digit 0:** `1 1 1 1 1 1 0`
- **Digit 1:** `0 1 1 0 0 0 0`
- **Digit 2:** `1 1 0 1 1 0 1`
- **Digit 3:** `1 1 1 1 0 0 1`
- **Digit 4:** `0 1 1 0 0 1 1`
- **Digit 5:** `1 0 1 1 0 1 1`
- **Digit 6:** `1 0 1 1 1 1 1`
- **Digit 7:** `1 1 1 0 0 0 0`
- **Digit 8:** `1 1 1 1 1 1 1`
- **Digit 9:** `1 1 1 1 0 1 1`

---

## ❓ Common Issues

### "torch not found"
```bash
pip3 install torch
```

### "best_model.pth not found"
- Train the model first (Option 1)

### Program won't run
```bash
# Check Python version (need 3.7+)
python3 --version

# Reinstall packages
pip3 install -r requirements.txt --force-reinstall
```

---

## 📚 Need Help?

- **Full Guide:** Read [README.md](README.md)
- **Menu Option 5:** Learning resources built into the program
- **Experiment:** Try different patterns and see what happens!

---

## 🎯 Your First Session Should Look Like This:

```bash
# 1. Install
$ pip3 install -r requirements.txt
[...packages installing...]

# 2. Run
$ python3 main.py

# 3. Choose Option 1 (Train)
Choose an option: 1
[Training starts...]
✓ Training complete!
🎯 Final Test Accuracy: 98.00%

# 4. Choose Option 3 (Predict)
Choose an option: 3
Enter pattern: 1 1 1 1 1 1 0
🎯 Prediction: 0
📊 Confidence: 99.8%
```

**That's it! You just trained your first neural network!** 🎉

---

**Now explore the other options and have fun learning!**
