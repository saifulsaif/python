from parent import Car
from radious import Circle
from ractangle import Rectangle


class TextBook():

 def print_book_title(self, book_title):
    print(book_title)



def main():
 # object text_book is the instant of the class TextBook():
 text_book = TextBook()
 custom_text = 'Its working'
 book_title = "Programming with Python and " + custom_text
 # calling print_book_title() method
 text_book.print_book_title(book_title)



my_car = Car("Toyota", "Corolla")
info = my_car.informations()
print(info)



circle = Circle("Red", 5)
print(f"Circle Area: {circle.area():.2f}")
print(f"Circle Perimeter: {circle.perimeter():.2f}\n")


rectangle = Rectangle("Blue", 10, 6)
print(f"Rectangle Area: {rectangle.area():.2f}")
print(f"Rectangle Perimeter: {rectangle.perimeter():.2f}")



if __name__ == '__main__':
 main()