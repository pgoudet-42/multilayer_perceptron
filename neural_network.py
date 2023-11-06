#!/usr/bin/python3

import model
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from load_data import getData

def neuralNetwork(X, y, hidden_layers=(30,30,30), learning_rate=0.1, n_iter=1000):
    np.random.seed(0)
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parametres = model.initialisation(dimensions)
    # for k,v in parametres.items():
    #     print(k, v.shape)
    
    train_loss = []
    train_acc = []
    
    for i in tqdm(range(n_iter)):
        activations = model.forwardPropagation(X, parametres)
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
    plt.show()
    
if __name__ == "__main__":
    training_set, y_training, test_set, y_test = getData("data.csv")
    neuralNetwork(training_set, y_training)
    exit(1)