# 1. Lists: Ordered, changeable, and allows duplicate members.
fruits = ["apple", "banana", "cherry"]

house_types = ["Building", "appartments", "Bungalow", "Cottage", "Villa"]


house_types.append("Dream home")
house_types.remove("Bungalow")
house_types.insert(2, "Penthouse")
house_types.pop()
house_types.sort()
house_types.reverse()
house_types.count("Villa")
house_types.index("Villa")
house_types.copy()
house_types.clear()
house_types.extend(["Villa", "Penthouse"])

print(house_types.reverse())

print(f"the house types are {house_types}")
print(type(house_types))

for house in house_types:
    print(f"the house name is {house}")



# 2. Tuples: Ordered and unchangeable.
coordinates = (10, 20)

print(f"Tuples (Unchangeable): {coordinates}")

# 3. Sets: Unordered, unchangeable*, and unindexed. No duplicate members.
colors = {"red", "green", "blue", "red", "red"} # "red" is a duplicate

print(f"Sets (No Duplicates): {colors}")

# 4. Dictionaries: Ordered** and changeable. No duplicate members (keys).
student = {
    "name": "Saiful",
    "course": "AI Specialist",
    "year": 2024,
    "favorite_number": 7
}
print(f"Dictionary: {student}")
print(f"Student Passing Year: {student['year']}")

# --- Practice Exercise ---
# Task: 
# 1. Create a list of your favorite coding languages.
# 2. Add 'Python' to that list if it's not already there.
# 3. Create a dictionary for a 'Car' with brands, model, and year.
# 4. Print the car's model.

print("\n--- Practice Exercise Output ---")
# Write your exercise code below
