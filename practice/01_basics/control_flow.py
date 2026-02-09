# 1. If-Else: Decision making
age = 18
if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")

# 2. For Loops: Iterating over a sequence
print("\nCounting to 5 using for loop:")
for i in range(1, 6):
    print(f"Number: {i}")

fruits = ["apple", "banana", "cherry"]
print("\nIterating through a list:")
for fruit in fruits:
    print(f"I like {fruit}")

# 3. While Loops: Executing as long as a condition is true
print("\nWhile loop countdown:")
count = 3
while count > 0:
    print(count)
    count -= 1
print("Blast off!")



count = 1
while count <= 10:
    print(count * 2)
    if count >  10:
        print("The number is ---- 6")
    else:
        print("The number is not 6")    
    count += 2
print("The loop run successfully")




# --- Practice Exercise ---
# Task:
# 1. Create a variable 'temperature'.
# 2. Write an if-elif-else block:
#    - If temp > 30: "It's hot"
#    - If temp > 20: "It's pleasant"
#    - Else: "It's cold"
# 3. Use a for loop to print even numbers from 2 to 10.

print("\n--- Practice Exercise Output ---")
# Write your exercise code below
