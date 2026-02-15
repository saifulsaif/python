# 01: List Comprehensions
# List comprehensions provide a concise way to create lists.
# Syntax: [expression for item in iterable if condition == True]

# 1. Basic List Comprehension
# Task: Create a list of squares for numbers 0-9
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")

# 2. List Comprehension with Condition
# Task: Create a list of even numbers from 0-19
evens = [x for x in range(20) if x % 2 == 0]
print(f"Evens: {evens}")

# 3. List Comprehension with if-else
# Task: Label numbers as 'Even' or 'Odd'
labels = ["Even" if x % 2 == 0 else "Odd" for x in range(10)]
print(f"Labels: {labels}")

# 4. Nested List Comprehension
# Task: Flatten a list of lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print(f"Flattened: {flattened}")

# Practice Task:
# Given a list of names, create a new list containing only names that start with 'A' and make them uppercase.
names = ["Alice", "Bob", "Charlie", "Anna", "David", "Alex"]
# Your code here:
# result = [...]
