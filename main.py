#!/usr/bin/python3

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from load_data import getData
import warnings
from perceptron import Perceptron
from layer import Layer
from sys import argv
import random

#suppress warnings
warnings.filterwarnings('ignore')

def trainModel(X, y):
    perceptron = Perceptron([Layer(X.shape[0], activation='sigmoid'),
                            Layer(24, activation='sigmoid'),
                            Layer(24, activation='sigmoid'),
                            Layer(24, activation='sigmoid'),
                            Layer(y.shape[0], activation='softmax')])
    params = perceptron.fit(X, y, learning_rate=0.0314, epochs=84)

    return (perceptron, params)


if __name__ == "__main__":
    # random.seed(123)
    X_train, y_train, X_test, y_test = getData("data.csv")

    # X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=0)
    # X, y = make_blobs(n_samples=300, centers=2, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))
    X_test = X_test.T
    test_y = y_test.reshape((1, y_test.shape[0]))

    perceptron, parametres = trainModel(X_train, y_train)
    if len(argv) > 1 and argv[1] == "--predict":
        y_pred = perceptron.predict(X_test, parametres)
        accuracy = accuracy_score(test_y.flatten(), y_pred.flatten())
        print("Accuracy Prediction:", accuracy)
    exit(0)