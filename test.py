#!/usr/bin/python3

from sklearn.neural_network import MLPClassifier
from load_data import getData

def test_func(X_train, y_train):

# Création d'un modèle de perceptron multicouche avec 3 couches cachées de 50, 30 et 10 neurones respectivement
    model = MLPClassifier(hidden_layer_sizes=(50, 30, 10))

# Supposons que X_train et y_train sont vos données d'entraînement
    model.fit(X_train, y_train)

# Accéder aux dimensions des couches
    dimensions_couches = [coef.shape for coef in model.coefs_]

    print("Dimensions des couches : ", dimensions_couches)

if __name__ == "__main__":
    train_X, train_y, test_X, test_y = getData("data.csv")
    test_func(train_X, train_y)
