import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

data = np.array(1.0)
print(data.ndim)  # Output: 0
x = Variable(data)
print(x.data)  # Output: 1.0

data = np.array(2.0)
print(data.ndim)  # Output: 0
x = Variable(data)
print(x.data)  # Output: 2.0
