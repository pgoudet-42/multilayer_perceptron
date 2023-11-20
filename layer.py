import numpy as np
from activations import softmax, sigmoid, relu

class Layer():
    def __init__(self, dimension: int, activation='sigmoid') -> None:
        functions = [softmax, sigmoid, relu]
        functions_name = ["softmax", "sigmoid", "relu"]
        index = functions_name.index(activation)
        self.dimension = dimension
        self.activation =functions[index]
    

    