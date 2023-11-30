import numpy as np
from activations import log_loss
from layer import Layer
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from layer import Layer

def display(training_history):
      plt.figure(figsize=(12, 4))
      plt.subplot(1, 2, 1)
      plt.plot(training_history[:, 0], label='train loss')
      plt.legend()
      plt.subplot(1, 2, 2)
      plt.plot(training_history[:, 2], label='train acc')
      plt.legend()
      plt.show()
    

def loadParams(file):
    parametres = {}
    keys = []
    values = []
    with open(file, "r") as f:
        lines = f.readlines()
        vals = []
        for line in lines:
            if line[0] == "(":
                if len(vals) != 0:
                    values.append(np.array(vals))
                    vals = []
                keys.append(line[1:-2])
            else:
                v = line[:-2].split(" ")
                vals.append(np.array(v).astype(float))
        values.append(np.array(vals))

    for i, k in enumerate(keys):
        parametres[k] = values[i]
    return parametres



def saveParams(parametres):
    with open("params.txt", "w") as f:
        for key, val in parametres.items():
            print("(" + key + ")", file=f)
            for v in val:
                for e in v:
                    print(e, file=f, end=" ")
                print("", file=f)

class Perceptron():
    def __init__(self, input_layer: int, output_layer: int, layers: tuple, activation: str = "sigmoid") -> None:
        self.nb_layers = layers[1] + 2
        self.layers = [Layer(input_layer, activation=activation)]
        self.layers.extend([Layer(layers[0], activation=activation) for i in range(layers[1])])
        self.layers.append(Layer(output_layer, activation='softmax'))
        self.dimensions = [obj.dimension for obj in self.layers]
    
    def initialisation(self):
        parametres = {}
        C = len(self.dimensions)
        for c in range(1, C):
            parametres['W' + str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c - 1])
            parametres['b' + str(c)] = np.random.randn(self.dimensions[c], 1)
        return parametres


    def forwardPropagation(self, X, parametres):
        activations = {'A0': X}
        C = len(parametres) // 2

        for c in range(1, C + 1):
            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            activations['A' + str(c)] = self.layers[c - 1].activation(Z)
        return activations
    
    def backPropagation(self, y, parametres, activations):
        m = y.shape[1]
        C = len(parametres) // 2
        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
        return gradients

    def update(self, gradients, parametres, learning_rate):
        C = len(parametres) // 2
        for c in range(1, C + 1):
            parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
            parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]
        return parametres

    def fit(self, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=100):
        parametres = self.initialisation()
        training_history = np.zeros((int(epochs), 3))
        C = len(parametres) // 2

        for i in range(epochs):
            activations = self.forwardPropagation(X_train, parametres)
            activations_test = self.forwardPropagation(X_test, parametres)
            gradients = self.backPropagation(y_train, parametres, activations)
            parametres = self.update(gradients, parametres, learning_rate)
            Af = activations['A' + str(C)]
            Af_test = activations_test['A' + str(C)]
            
            training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))
            training_history[i, 1] = (log_loss(y_test.flatten(), Af_test.flatten()))

            activations = activations = self.forwardPropagation(X_train, parametres)
            C = len(parametres) // 2
            y_pred = activations['A' + str(C)] >= 0.5
            training_history[i, 2] = (accuracy_score(y_pred.flatten(), y_train.flatten())
                                      )
            print(f"epoch {i}/{epochs - 1:.4f} - loss: {training_history[i, 0]:.4f}- val_loss: {training_history[i, 1]:.4f}")

        saveParams(parametres)
        display(training_history)
        return (parametres)

    def predict(self, X):
        parametres = loadParams("params.txt")
        activations = self.forwardPropagation(X, parametres)
        C = len(parametres) // 2
        Af = activations['A' + str(C)]
        return Af >= 0.5
