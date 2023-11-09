#!/usr/bin/python3

import model
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from load_data import getData

def neuralNetwork(X, y, hidden_layers=(30,30), learning_rate=0.1, n_iter=10):
    np.random.seed(0)
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[1])
    dimensions.append(y.shape[0])
    parametres = model.initialisation([2, 30, 30, 1])
    for k,v in parametres.items():
        print(k, v.shape)
    activations = model.forwardPropagation(X, parametres)
    for k,v in activations.items():
        print(k, v.shape)
    train_loss = []
    train_acc = []
    
    for i in tqdm(range(n_iter)):
        activations = model.forwardPropagation(X, parametres)
        # for k,v in parametres.items():
        #   print(k, v.shape)
        gradients = model.backPropagation(y, activations, parametres)
        parametres = model.update(gradients, parametres, learning_rate)
        
        if i % 10 == 0:
            C = len(parametres ) // 2
            train_loss.append(model.log_loss(activations['A' + str(C)], y))
            y_pred = model.predict(X, parametres)
            current_accuracy = accuracy_score(y[:, 0], y_pred[:, 0])
            train_acc.append(current_accuracy)


    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label="train loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_acc, label="train accuracy")
    plt.legend()
    # plt.show()
    
if __name__ == "__main__":
    # train_X, train_y, test_X, test_y = getData("data.csv")
    # print(train_X.shape)
    # print(train_y.shape)
    # print(test_X.shape)
    # print(test_y.shape)
    X, y= make_blobs(n_samples=100, n_features=2, centers=2, random_state=8)
    y = y.reshape((y.shape[0], 1))
    neuralNetwork(X, y)
    exit(1)