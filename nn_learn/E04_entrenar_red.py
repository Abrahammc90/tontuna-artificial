"""
Archivo: nn_learn/E04_entrenar_red.py
Objetivo: Entrenar la red neuronal con los datos preparados y evaluar su rendimiento.

En este archivo aprenderás a:
1. Cargar la red neuronal y los datos preparados.
2. Entrenar la red usando descenso de gradiente estocástico (SGD).
3. Evaluar el rendimiento en los datos de prueba.

Puedes ejecutar este archivo para entrenar tu red y ver los resultados.
"""

import numpy as np
from nn_copiloto import Network
import pickle
import gzip
import os
import sys

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Cargar los datos igual que antes
data_file = sys.argv[1]
with open(data_file, 'rb') as f:
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
training_results = [vectorized_result(y) for y in training_data[1]]
training_data_prepared = list(zip(training_inputs, training_results))

validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
validation_data_prepared = list(zip(validation_inputs, validation_data[1]))

test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
test_data_prepared = list(zip(test_inputs, test_data[1]))

# Crear la red neuronal
net = Network([784, 30, 10])

# Entrenar la red
# epochs=10, mini_batch_size=10, eta=3.0 (puedes ajustar estos valores)
net.SGD(training_data_prepared, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data_prepared)

# Al finalizar, verás cuántos aciertos tuvo la red en los datos de prueba en cada época.
# Puedes modificar los hiperparámetros y observar cómo cambia el rendimiento.
