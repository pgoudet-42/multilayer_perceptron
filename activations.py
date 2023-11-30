import numpy as np
from scipy.special import xlogy


def identity(X):
    return (X)

def softmax(X):
    e_x = np.exp(X)
    return e_x / np.sum(e_x)
    
def sigmoid(X):
    return 1 / (1 + np.exp(-X))
    
def relu(a):
    return np.maximum(0, a)

def log_loss(y_true, y_pred, *, eps="auto", normalize=True):
    n = len(y_true)
    loss = -(xlogy(y_true, y_pred) + xlogy(1 - y_true, 1 - y_pred)).sum()
    loss /= n
    return loss

def binary_crossentropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    loss = np.mean(loss)
    return loss