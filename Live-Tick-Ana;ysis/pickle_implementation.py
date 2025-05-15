import pickle


class MyClass:
    def __init__(self, initial_value):
        self.value = initial_value

    def calculate_and_store(self):
        self.calculated_value = self.value * 2

    def use_calculated_value(self):
        if hasattr(self, "calculated_value"):
            return self.calculated_value
        else:
            return "Calculation not performed yet"


# Version 1: Pickle the object before calculate_and_store
obj1 = MyClass(10)
with open("my_object_v1.pickle", "wb") as f:
    pickle.dump(obj1, f)

# Version 2: Calculate and then pickle
obj2 = MyClass(20)
obj2.calculate_and_store()
with open("my_object_v2.pickle", "wb") as f:
    pickle.dump(obj2, f)

# Load version 1
with open("my_object_v1.pickle", "rb") as f:
    loaded_obj1 = pickle.load(f)
print(loaded_obj1.use_calculated_value())  # Output: Calculation not performed yet

# Load version 2
with open("my_object_v2.pickle", "rb") as f:
    loaded_obj2 = pickle.load(f)
print(loaded_obj2.use_calculated_value())  # Output: 40
