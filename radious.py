import math

class Circle:
    def __init__(self, color, radius):
        self.color = color
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

    def __str__(self):
        return f"Circle(Color: {self.color}, Radius: {self.radius})"