"""
Archivo: nn_learn/04_entrenar_red.py
Objetivo: Entrenar la red neuronal con los datos preparados y evaluar su rendimiento.

En este archivo aprenderás a:
1. Cargar la red neuronal y los datos preparados.
2. Entrenar la red usando descenso de gradiente estocástico (SGD).
3. Evaluar el rendimiento en los datos de prueba.

Puedes ejecutar este archivo para entrenar tu red y ver los resultados.
"""

import numpy as np
from nn_learn.02_red_neuronal import Network
import pickle
import gzip
import os

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Cargar los datos igual que antes
input_dir = os.path.join(os.path.dirname(__file__), '..', 'input-files')
data_file = None
for fname in ['mnist.pkl', 'mnist.pkl.gz']:
    path = os.path.join(input_dir, fname)
    if os.path.exists(path):
        data_file = path
        break
if data_file is None:
    raise FileNotFoundError('No se encontró mnist.pkl ni mnist.pkl.gz en input-files')

if data_file.endswith('.gz'):
    with gzip.open(data_file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
else:
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
