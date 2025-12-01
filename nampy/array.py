import numpy as np
np.random.seed(42)

arr1 = a = np.array([1, 2, 3, 4, 5])


print(f"a Array: {a}")


arr1[4] = 10
print(f"arr1 Array: {arr1}")


arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D Array:\n{arr2}")
print(f"Shape: {arr2.shape}, Dimensions: {arr2.ndim}")


zeros = np.zeros((3,4))
print(zeros)


ones = np.ones((3,3)) * 7
print(ones)


random_arr = np.random.rand(2, 5) 
print(random_arr)