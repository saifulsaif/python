# 1. Basic String Operations
message = "Python for AI"
print(f"Original: {message}")

# 2. String Methods
print(f"Uppercase: {message.upper()}")
print(f"Lowercase: {message.lower()}")
print(f"Length: {len(message)}")
print(f"Check if 'AI' is in message: {'AI' in message}")

# 3. Slicing
# [start:end:step]
print(f"First 6 characters: {message[0:6]}")
print(f"Last word: {message[-2:]}")

# 4. Splitting and Joining
text = "apple,banana,cherry"
fruits_list = text.split(",")
print(f"Split string into list: {fruits_list}")

# --- Practice Exercise ---
# Task:
# 1. Create a variable 'sentence' = "I am learning Python for AI".
# 2. Print the length of the sentence.
# 3. Convert the total sentence to uppercase.
# 4. Replace "AI" with "Machine Learning".
# 5. Reverse the string using slicing (Hint: [::-1]).

print("\n--- Practice Exercise Output ---")
# Write your exercise code below
