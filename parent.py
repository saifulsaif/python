# inheritance.py

class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def informations(self):
        return f"Car Brand: {self.brand}, Model: {self.model}"
