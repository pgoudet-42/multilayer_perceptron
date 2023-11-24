import numpy as np
from activations import softmax, sigmoid, relu, identity

class Layer():
    def __init__(self, dimension: int, activation='sigmoid') -> None:
        functions = [softmax, sigmoid, relu, identity]
        functions_name = ["softmax", "sigmoid"]
        index = functions_name.index(activation)
        self.dimension = dimension
        self.activation =functions[index]
    

    