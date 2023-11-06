#!/usr/bin/python3

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from load_data import getData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from test import test_prog

def display_sets(data, explained_variance_ratio):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:,0], data[:,1])
    plt.xlabel(f'{round(explained_variance_ratio[0], 2) * 100}%')
    plt.ylabel(f'{round(explained_variance_ratio[1], 2) * 100}%')
    plt.show()

def afd(data, etiquette):
    # pca = PCA(n_components=2)
    # cp_benign = pca.fit(data)
    # components = range(1, len(pca.explained_variance_) + 1)
    lda = LinearDiscriminantAnalysis()
    print(data, etiquette)
    lda.fit(data, etiquette)
    rndt = lda.transform(data)
    print(rndt.shape)
    plt.plot(rndt, 'r+')
    plt.xlabel("Axe discriminant")
    plt.ylabel("Classe")
    plt.yticks([1, 2])
    plt.show()

def acp(data, n_components):
    pca = PCA(n_components=n_components)
    pc1 = pca.fit_transform(data)
    return pc1

def custom_acp(X, n_components):
    mean = np.mean(X, axis=0)
    centered_X = X - mean
    # Calcul de la matrice de covariance
    cov_matrix = np.cov(X, rowvar=False)
    # Calcul des valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Triez les vecteurs propres en fonction des valeurs propres décroissantes
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    # Sélectionnez les premières n_components composantes principales
    top_eigenvectors = eigenvectors[:, :n_components]
    # Transformez les données originales dans l'espace des composantes principales
    transformed_X = np.dot(X, top_eigenvectors)
    return transformed_X

def visualization():
    training_set, _ = getData("data.csv")
    
    etiquettes = training_set[:, 1]
    for i, _ in enumerate(etiquettes):  etiquettes[i] = 0 if etiquettes[i] == 'B' else  1
    etiquettes = np.array(etiquettes)
    
    training_set = np.delete(training_set, (0, 1), axis=1)
    training_set = training_set.astype(float)

    
    scaler = StandardScaler()
    scaler.fit_transform(training_set)
    n_components = 10
    pc1 = custom_acp(training_set, n_components)
    display_sets(pc1, (0,0))
    pc1 = acp(training_set, n_components)
    display_sets(pc1, (0,0))
    test_prog(pc1, etiquettes)
    

if __name__ == "__main__":
    exit(visualization())