#!/usr/bin/python3

import model
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles

from load_data import getData

import warnings

#suppress warnings
warnings.filterwarnings('ignore')

def neuralNetwork(X, y, hidden_layers=(32,32), learning_rate=0.1, n_iter=1000):
    np.random.seed(0)
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parametres = model.initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = model.forwardPropagation(X, parametres)
        gradients = model.backPropagation(y, parametres, activations)
        parametres = model.update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (model.log_loss(y.flatten(), Af.flatten()))
        y_pred = model.predict(X, parametres)
        training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))
    
    print(y.flatten())
    print(y_pred.flatten())
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    X, y, test_X, test_y = getData("data.csv")
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)
    # X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=0)
    # X, y = make_blobs(n_samples=300, centers=2, random_state=42)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    # print('dimensions de X:', X.shape)
    # print('dimensions de y:', y.shape)
    # y = y.reshape((y.shape[0], 1))
    # neuralNetwork(train_X, train_y)
    neuralNetwork(X, y)
    exit(1)