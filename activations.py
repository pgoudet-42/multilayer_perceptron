import numpy as np


def softmax(X):
    e_x = np.exp(X)
    return e_x / np.sum(e_x)
    
def sigmoid(X):
    return 1 / (1 + np.exp(-X))
    
def relu(a):
    return np.maximum(0, a)
