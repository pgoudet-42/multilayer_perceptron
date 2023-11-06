import numpy as np

class PerceptronMulticouche:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialisation des poids pour les couches cachées et de sortie
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        # Initialisation des biais pour les couches cachées et de sortie
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, inputs):
        # Calcul de la sortie de la couche cachée
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)

        # Calcul de la sortie finale
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)

        return hidden_output, output

    def train(self, inputs, targets, learning_rate):
        hidden_output, output = self.feed_forward(inputs)

        # Calcul de l'erreur
        output_error = targets - output

        # Calcul des gradients pour les poids de la couche de sortie
        output_delta = output_error * self.sigmoid_derivative(output)
        weight_output_gradient = np.dot(hidden_output.reshape(-1, 1), output_delta)
        bias_output_gradient = output_delta

        # Calcul de l'erreur pour la couche cachée
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)

        # Calcul des gradients pour les poids de la couche cachée
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)
        weight_hidden_gradient = np.dot(inputs.reshape(-1, 1), hidden_delta)
        bias_hidden_gradient = hidden_delta

        # Mise à jour des poids et des biais
        self.weights_hidden_output += learning_rate * weight_output_gradient
        self.bias_output += learning_rate * bias_output_gradient
        self.weights_input_hidden += learning_rate * weight_hidden_gradient
        self.bias_hidden += learning_rate * bias_hidden_gradient

    def predict(self, inputs):
        _, output = self.feed_forward(inputs)
        return output

# Exemple d'utilisation
if __name__ == "__main__":
    # Données d'entraînement
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Création du modèle MLP
    mlp = PerceptronMulticouche(input_size=2, hidden_size=4, output_size=1)

    # Entraînement du modèle
    epochs = 10000
    learning_rate = 0.1
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            mlp.train(X[i], y[i], learning_rate)

    # Prédiction
    for i in range(X.shape[0]):
        prediction = mlp.predict(X[i])
        print(f"Entrée: {X[i]}, Prédiction: {prediction}")
