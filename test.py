#!/usr/bin/python3

from sklearn.neural_network import MLPClassifier
from load_data import getData
from sklearn.datasets import make_blobs, make_circles

X, y, test_X, test_y = getData("data.csv")
# X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=0)
# X, y = make_blobs(n_samples=300, centers=2, random_state=42)
# X = X.T
# y = y.reshape((1, y.shape[0]))
mlp = MLPClassifier(hidden_layer_sizes=(100,3), learning_rate="constant", learning_rate_init=0.01, activation="logistic", max_iter=500)
mlp.fit(X,y)
print("cost:", mlp.loss_)
mlp.predict(test_X)
print(mlp.score(test_X, test_y))