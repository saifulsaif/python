# 1. Defining a basic function
def greet():
    print("Welcome to Python Functions!")

greet() # Calling the function

# 2. Function with arguments
def greet_user(name):
    print(f"Hello, {name}!")

greet_user("Saiful")

# 3. Function with return value
def square(number):
    return number * number

result = square(5)
print(f"The square of 5 is: {result}")

# 4. Keyword and Default arguments
def describe_pet(pet_name, animal_type="dog"):
    print(f"I have a {animal_type} named {pet_name}.")

describe_pet(pet_name="Buddy") # Uses default animal_type
describe_pet("Whiskers", "cat") # Overwrites default

# --- Practice Exercise ---
# Task:
# 1. Create a function called 'calculate_area' that takes 'length' and 'width'.
# 2. It should return the area (length * width).
# 3. Call the function and print the result.
# 4. Create a function that takes a name and prints it 3 times using a loop.

print("\n--- Practice Exercise Output ---")
# Write your exercise code below
