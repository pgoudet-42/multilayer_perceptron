import numpy as np
from activations import softmax, sigmoid, relu, log_loss
from layer import Layer
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from layer import Layer


def display(training_history):
      plt.figure(figsize=(12, 4))
      plt.subplot(1, 2, 1)
      plt.plot(training_history[:, 0], label='train loss')
      plt.legend()
      plt.subplot(1, 2, 2)
      plt.plot(training_history[:, 1], label='train acc')
      plt.legend()
      plt.show()

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

    def fit(self, data_train, data_valid, learning_rate=0.01, epochs=100):
        parametres = self.initialisation()
        training_history = np.zeros((int(epochs), 2))
        C = len(parametres) // 2

        for i in tqdm(range(epochs)):
            activations = self.forwardPropagation(data_train, parametres)
            gradients = self.backPropagation(data_valid, parametres, activations)
            parametres = self.update(gradients, parametres, learning_rate)
            Af = activations['A' + str(C)]

            training_history[i, 0] = (log_loss(data_valid.flatten(), Af.flatten()))
            y_pred = self.predict(data_train, parametres)
            training_history[i, 1] = (accuracy_score(data_valid.flatten(), y_pred.flatten()))
            print(f"epoch {i}/{epochs - 1:.4f} - loss: {training_history[i, 0]:.4f}- accuracy: {training_history[i, 1]:.4f}")

        print("Accuracy Training:", training_history[-1, 1])
        # display(training_history)
        return (parametres)

    def predict(self, X, parametres):
        activations = self.forwardPropagation(X, parametres)
        C = len(parametres) // 2
        Af = activations['A' + str(C)]
        return Af >= 0.5
