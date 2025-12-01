class Rectangle:
    def __init__(self, color, length, width):
        self.color = color
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * (self.length + self.width)

    def __str__(self):
        return f"Rectangle(Color: {self.color}, Length: {self.length}, Width: {self.width})"
