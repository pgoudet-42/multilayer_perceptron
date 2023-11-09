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

def splitData(data) -> tuple:
    training = []
    training_size = int(0.7 * len(data))
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

def getData(file: str):
    data: list = readCSV(file)
    training_set, test_set = splitData(data)
    
    etiquettes_training = training_set[:, 1]
    for i, _ in enumerate(etiquettes_training):  etiquettes_training[i] = 0 if etiquettes_training[i] == 'B' else  1
    etiquettes_training = np.array(etiquettes_training)
    etiquettes_training = etiquettes_training.astype(int)
    # etiquettes_training = etiquettes_training.reshape(-1, 1)
    
    etiquettes_test = test_set[:, 1]
    for i, _ in enumerate(etiquettes_test):  etiquettes_test[i] = 0 if etiquettes_test[i] == 'B' else  1
    etiquettes_test = np.array(etiquettes_test)
    etiquettes_test = etiquettes_test.astype(int)
    # etiquettes_test = etiquettes_test.reshape(-1, 1)
    
    training_set = np.delete(training_set, (0, 1), axis=1)
    training_set = training_set.astype(float)
    test_set = np.delete(test_set, (0, 1), axis=1)
    test_set = test_set.astype(float)


    scaler = StandardScaler()
    scaler.fit_transform(training_set)
    return (training_set, etiquettes_training, test_set, etiquettes_test)
    
    