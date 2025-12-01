import numpy as np

data = np.array([
    [10, 15, 20, 25, 30],
    [35, 40, 45, 50, 55],
    [60, 65, 70, 75, 80],
    [85, 90, 95, 40, 10]
])

print(data)

element = data[2][3]
#print(f"\n1. Element at [2, 3]: {element}")

#print(f"Element at [1, 2]: {data[2, 3]}")

values_gt_50 = data[ data > 50]


#print(f"\n5. Values > 50: {values_gt_50}")

print(f"\nFirst row: {data[1:2, 1:4]}")

print(f"Submatrix:\n{data[1:3, 1:3]}")

indices = [0, 1,2]
print(f"\nRows 0 and 2:\n{data[indices, :]}")