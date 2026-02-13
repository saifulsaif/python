# 03: Error Handling
# Handling exceptions gracefully is crucial for robust applications.

# 1. Basic Try-Except
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"Result: {result}")
except ValueError:
    print("Error: Please enter a valid integer.")
except ZeroDivisionError:
    print("Error: Cannot divide by zero.")

# 2. Finally and Else blocks
try:
    file = open("sample.txt", "w")
    file.write("Hello Errors!")
except IOError:
    print("Could not write to file.")
else:
    print("File written successfully.")
finally:
    if 'file' in locals():
        file.close()
        print("File closed.")

# 3. Raising Exceptions
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    return f"Age is {age}"

try:
    print(check_age(-5))
except ValueError as e:
    print(f"Caught an error: {e}")

# Practice Task:
# Write a function that takes a list of numbers and returns their average.
# Use try-except to handle cases where the list is empty (ZeroDivisionError) 
# or contains non-numeric values (TypeError).
