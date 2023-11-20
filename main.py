#!/usr/bin/python3

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from load_data import getData
import warnings
from perceptron import Perceptron
from layer import Layer

#suppress warnings
warnings.filterwarnings('ignore')

def trainModel(X, y):
    perceptron = Perceptron([Layer(30, activation='sigmoid'), 
                            Layer(24, activation='sigmoid'), 
                            Layer(24, activation='sigmoid'), 
                            Layer(1, activation='softmax')])
    perceptron.fit(X, y, learning_rate=0.01, epochs=1000)

if __name__ == "__main__":
    X, y, test_X, test_y = getData("data.csv")
    # X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=0)
    # X, y = make_blobs(n_samples=300, centers=2, random_state=42)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    trainModel(X, y)
    exit(1)