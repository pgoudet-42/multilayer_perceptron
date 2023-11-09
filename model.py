import numpy as np

def softmax(X):
    e_x = np.exp(X)
    return e_x / np.sum(e_x)

def log_loss(A, y):
    return -(1 / len(y)) * np.sum(y * np.log(A))

def initialisation(dim: list):
    params = {}
    C = len(dim)
    
    for  i in range(1, C):
        params["W" + str(i)] = np.random.rand(dim[i], dim[i - 1])
        params["b" + str(i)] = np.random.rand(dim[i], 1)
    
    return params

def forwardPropagation(X, params: dict):
    activations = {"A0": X.T}
    C = len(params) // 2
    
    for c in range(1, C + 1):
        print("c:", c)
        print("W" + str(c) + ":", params["W" + str(c)].shape)
        print("A" + str(c -1) + ":", activations["A" + str(c - 1)].shape)
        print("b" + str(c) + ":", params["b" + str(c)].shape)
        print("resultat:", params["W" + str(c)].dot(activations["A" + str(c - 1)]  + params["b" + str(c)] ) )

        Z = params["W" + str(c)].dot(activations["A" + str(c - 1)]) + params["b" + str(c)]
        activations["A" + str(c)] = 1 / (1 + np.exp(-Z))

    
    return activations

def backPropagation(y, activations, parametres):
    m = y.shape[1]
    C = len(parametres) // 2
    
    dZ = activations['A' + str(C)] - y
    gradients = {}
    
    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations["A" + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres["W" + str(c)].T, dZ) * activations["A" + str(c - 1)] * (1 - activations['A' + str(c - 1)])
        
    return (gradients)

def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2
    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]
    
    return (parametres)

def predict(X, parametres):
    activations = forwardPropagation(X, parametres)
    C = len(parametres) // 2
    Af = activations['A' + str(C)]
    print(Af)
    return Af >= 0.5