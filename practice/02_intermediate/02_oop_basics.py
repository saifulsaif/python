# 02: Object-Oriented Programming (OOP)
# OOP is essential for building complex systems and using AI libraries like PyTorch.

# 1. Defining a Class
class Robot:
    """A simple class to represent a Robot."""
    
    # Class Attribute
    species = "Artificial Intelligence"

    # Constructor (Initializer)
    def __init__(self, name, version):
        self.name = name          # Instance Attribute
        self.version = version    # Instance Attribute
        self.battery_level = 100  # Initial state

    # Instance Method
    def introduce(self):
        print(f"Hello, I am {self.name}, version {self.version}. Species: {self.species}")

    def charge(self, amount):
        self.battery_level = min(100, self.battery_level + amount)
        print(f"{self.name} charged to {self.battery_level}%")

# 2. Creating Objects (Instances)
robot1 = Robot("Alpha", 1.0)
robot1.introduce()
robot1.charge(20)

# 3. Inheritance
class AI_Robot(Robot):
    """A specialized robot with AI capabilities."""
    
    def __init__(self, name, version, brain_type):
        super().__init__(name, version) # Initialize parent class
        self.brain_type = brain_type

    # Overriding methods
    def introduce(self):
        print(f"I am {self.name}, an advanced {self.brain_type} powered AI.")

# Practice Task:
# 1. Create a 'Book' class with title, author, and pages.
# 2. Add a method 'is_long' that returns True if pages > 300.
# 3. Create an instance of 'Book' and test the method.
