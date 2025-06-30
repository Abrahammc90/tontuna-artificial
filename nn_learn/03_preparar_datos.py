"""
Archivo: nn_learn/03_preparar_datos.py
Objetivo: Preparar los datos para entrenar la red neuronal.

En este archivo aprenderás a transformar los datos de entrada y salida en el formato adecuado para la red.

Pasos:
1. Cargar los datos usando el script 01_cargar_datos.py.
2. Convertir las imágenes a vectores columna (shape (784, 1)).
3. Convertir las etiquetas a vectores one-hot (para la salida de la red).
4. Crear listas de tuplas (x, y) para entrenamiento, validación y prueba.

Puedes ejecutar este archivo para ver ejemplos de datos preparados.
"""

import numpy as np
import pickle
import gzip
import os

def vectorized_result(j):
    """Convierte una etiqueta (0-9) en un vector one-hot de 10 dimensiones."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Cargar los datos igual que en 01_cargar_datos.py
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

# Preparar los datos de entrenamiento
training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
training_results = [vectorized_result(y) for y in training_data[1]]
training_data_prepared = list(zip(training_inputs, training_results))

# Preparar los datos de validación y prueba
validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
validation_data_prepared = list(zip(validation_inputs, validation_data[1]))

test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
test_data_prepared = list(zip(test_inputs, test_data[1]))

print(f"Primer ejemplo de entrenamiento (input shape): {training_data_prepared[0][0].shape}")
print(f"Primer ejemplo de entrenamiento (output one-hot): {training_data_prepared[0][1].T}")
print(f"Primer ejemplo de validación (input shape): {validation_data_prepared[0][0].shape}")
print(f"Primer ejemplo de test (input shape): {test_data_prepared[0][0].shape}")

# Ahora puedes usar training_data_prepared, validation_data_prepared y test_data_prepared para entrenar tu red.
