# Lesson 01: Variables and Data Types
# In Python, we don't need to declare types. Python is dynamic!

# 1. Variables and Print
name = "Student"
age = 20
gpa = 3.8
is_learning_ai = True

full_name = "Md Saiful Islam"
age_of_saiful = 20
gpa_of_saiful = 3.8
is_learning_ai_of_saiful = True

print(f"Hello {name}!")
print(f"You are {age} years old and your GPA is {gpa}.")
print(f"Is learning AI? {is_learning_ai}")

# 2. Basic Arithmetic (Crucial for AI math)
a = 10
b = 3
print("\n--- Math Operations ---")
print(f"Addition: 10 + 3 = {a + b}")
print(f"Division (Float): 10 / 3 = {a / b}")
print(f"Division (Floor): 10 // 3 = {a // b} (Removes decimals)")
print(f"Power: 10^3 = {a ** b}")

print(f"Full Name: {full_name}")
print(f"Age: {age_of_saiful}")
print(f"GPA: {gpa_of_saiful}")
print(f"Is Learning AI? {is_learning_ai_of_saiful}")

print(f"this is the first test of my life { full_name }")
# 3. Dynamic Typing
x = 5
print(f"\nx is {x} (Type: {type(x)})")
x = "Now I am a string"
print(f"x is now '{x}' (Type: {type(x)})")


height = 40
width = 20
area = height * width
print(f"Area of the rectangle is {area}")

# EXERCISE: 
# Create a variable 'radius' with value 5
# Calculate the area of a circle (Area = 3.14 * r^2)
# Print the result.
