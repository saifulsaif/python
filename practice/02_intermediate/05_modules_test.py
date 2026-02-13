# 05: Modules and Packages
# Organizing code into modules helps maintainability and reuse.

# 1. Importing a custom module
import utils

welcome_msg = utils.greet_learner("AI Student")
print(welcome_msg)

# 2. Importing specific items
from utils import square_root, Calculator

print(f"Square root of 16 is: {square_root(16)}")

calc = Calculator()
print(f"5 + 7 = {calc.add(5, 7)}")

# 3. Importing standard library modules
import math
import os

print(f"Current directory: {os.getcwd()}")
print(f"Pi is approximately: {math.pi}")

# Practice Task:
# 1. Create a new module named 'geometry.py' in the same folder.
# 2. Add functions to calculate the area of a circle and a rectangle.
# 3. Import them here and use them.
