#!/usr/bin/python3

from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score
from load_data import getData
import warnings
from perceptron import Perceptron
from sys import argv
import argparse 

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de démonstration avec argparse")
    parser.add_argument('-t', '--train', action='store_true', help='fichier a charger')
    parser.add_argument('-f', '--fichier', type=str, help='fichier a charger')
    parser.add_argument('-p', '--predict', action='store_true', help='faire la prediction sur un set de test')
    args = parser.parse_args()

    if not args.fichier:
        print("Error: please give a data file to test/train")
        exit(1)

    X_train, y_train, X_test, y_test = getData(args.fichier, 0.7)

    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))
    X_test = X_test.T
    test_y = y_test.reshape((1, y_test.shape[0]))

    perceptron = Perceptron(input_layer=X_train.shape[0], output_layer=y_train.shape[0], layers=(100,2), activation="sigmoid")
    if args.train:
        perceptron.fit(X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=1500)

    if args.predict:
        y_pred = perceptron.predict(X_test)
        with open("results.txt", "w+")as f:
            for e in y_pred.flatten():
                print(e, file=f)
        accuracy = accuracy_score(test_y.flatten(), y_pred.flatten())
        print("Accuracy Prediction:", accuracy)
    exit(0)