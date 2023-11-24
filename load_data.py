import numpy as np
from random import randint
import csv
from sklearn.preprocessing import StandardScaler

CATEGORIES = ["ID", "class"] + [f"x{i}" for i in range(1, 31)]

def readCSV(file:str) -> list:
    data = []
    with open(file) as f:
        csvreader = csv.reader(f)
        try:
            for i,e in enumerate(csvreader):
                data.append(e)
                for j,c in enumerate(CATEGORIES):
                    if data[i][j] == None:
                        print(f"error at data[{i}]: ", data[i])
                        break
        except Exception as e:
            print("error:", e.args)
    return data

def splitData(data, percent_of_train) -> tuple:
    training = []
    training_size = int(percent_of_train * len(data))
    indexs = set()
    
    while len(indexs) < training_size:
        indexs.add(randint(0, training_size - 1))
    indexs = list(indexs)
    indexs.sort()
    indexs = indexs[::-1]
    
    for i in range(training_size):
        training.append(data.pop(indexs[i]))
    
    test = np.array(data)
    training = np.array(training)
    return (training, test)

def getEtiquettes(set):
    etiquettes = set[:, 1]
    for i, _ in enumerate(etiquettes):  etiquettes[i] = 0 if etiquettes[i] == 'B' else  1
    etiquettes = np.array(etiquettes)
    etiquettes = etiquettes.astype(int)
    return etiquettes

def getData(file: str, percent_of_train: float):
    try:
        data: list = readCSV(file)
        training_set, test_set = splitData(data, percent_of_train)
        
        etiquettes_training = getEtiquettes(training_set)
        etiquettes_test = getEtiquettes(test_set)
        
        training_set = np.delete(training_set, (0, 1), axis=1)
        training_set = training_set.astype(float)
        test_set = np.delete(test_set, (0, 1), axis=1)
        test_set = test_set.astype(float)

        scaler = StandardScaler()
        scaler.fit_transform(training_set)
        print("x_train shape:", training_set.shape)
        print("x_valid shape:", test_set.shape)
    except Exception as e:
        print(e.args)
        exit(1)
    return (training_set, etiquettes_training, test_set, etiquettes_test)
    
    