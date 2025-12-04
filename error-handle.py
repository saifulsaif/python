
def show_message(message):
    try:
        print(message)
        return message
    except Exception:
        print("Something went worng")
        return None

        


# Example 1: Division by Zero Exception
def divide_numbers(a, b):
    """Divides two numbers with error handling"""
    try:
        result = a / b
        show_message(f"Division successful: {a} / {b} = {result}")

    except ZeroDivisionError:
        show_message(f"Error: Cannot divide {a} by zero!")
        return None
    except TypeError:
        show_message(f"Error: Invalid data types - both arguments must be numbers")
        return None


# Example 2: List Index Error
def get_element_at_index(my_list, index):
    """Retrieves element from list with error handling"""
    try:
        element = my_list[index]
        print(f"Element at index {index}: {element}")
        return element
    except IndexError:
        print(f"Error: Index {index} is out of range for list of size {len(my_list)}")
        return None
    except TypeError:
        print(f"Error: Index must be an integer")
        return None


# ====================
# TEST DATA AND EXECUTION
# ====================

def run_all_tests():
    """Runs all error handling examples with test data"""

    print("=" * 60)
    print("PYTHON ERROR HANDLING EXAMPLES")
    print("=" * 60)

    # Test 1: Division
    print("\n--- Test 1: Division Operations ---")
    divide_numbers(10, 2)
    divide_numbers(10, 0)  # ZeroDivisionError
    divide_numbers(10, "2")  # TypeError

    # Test 2: List Index Access
    print("\n--- Test 2: List Index Access ---")
    fruits = ["apple", "banana", "cherry", "date"]
    get_element_at_index(fruits, 1)
    get_element_at_index(fruits, 10)  # IndexError
    get_element_at_index(fruits, "2")  # TypeError

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


# Run all tests when script is executed directly
if __name__ == "__main__":
    run_all_tests()
