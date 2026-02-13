# 04: File Handling
# Reading and writing data is a fundamental skill, especially for AI data processing.

# 1. Writing to a file
with open("practice_file.txt", "w") as f:
    f.write("Line 1: Learning Python for AI\n")
    f.write("Line 2: Phase 2 is about productivity\n")

# 2. Reading from a file
print("--- Reading entire file ---")
with open("practice_file.txt", "r") as f:
    content = f.read()
    print(content)

# 3. Reading line by line
print("--- Reading line by line ---")
with open("practice_file.txt", "r") as f:
    for line in f:
        print(f"Read: {line.strip()}")

# 4. Working with JSON (Super important for AI APIs)
import json

data = {
    "name": "Practice Robot",
    "tasks": ["Learning", "Coding", "Processing"],
    "active": True
}

# Save as JSON
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

# Read JSON
with open("data.json", "r") as f:
    loaded_data = json.load(f)
    print(f"Loaded JSON: {loaded_data['name']}")

# Practice Task:
# 1. Create a dictionary with student names and their scores.
# 2. Save it to a file named 'scores.json'.
# 3. Read it back and print the name of the student with the highest score.
