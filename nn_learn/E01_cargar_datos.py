"""
Archivo: nn_learn/01_cargar_datos.py
Objetivo: Cargar y explorar los datos de entrada para la red neuronal desde input-files.

Este es el primer paso para crear una red neuronal desde cero. Aquí aprenderás a cargar los datos y a inspeccionarlos para entender su estructura.

Pasos:
1. Cargar el archivo MNIST (input-files/mnist.pkl o mnist.pkl.gz).
2. Mostrar la cantidad de datos de entrenamiento, validación y prueba.
3. Visualizar un ejemplo de imagen y su etiqueta.

Puedes ejecutar este archivo con:
    python nn_learn/01_cargar_datos.py

Si todo funciona bien, deberías ver información sobre los datos y una imagen de ejemplo en consola.
"""

import pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Buscar el archivo de datos en input-files
input_dir = os.path.join(os.path.dirname(__file__), '..', 'input-files')
data_file = None
for fname in ['mnist.pkl', 'mnist.pkl.gz']:
    path = os.path.join(input_dir, fname)
    if os.path.exists(path):
        data_file = path
        break
if data_file is None:
    raise FileNotFoundError('No se encontró mnist.pkl ni mnist.pkl.gz en input-files')

# 2. Cargar los datos
if data_file.endswith('.gz'):
    with gzip.open(data_file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
else:
    with open(data_file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

print(f"Datos de entrenamiento: {len(training_data[0])} ejemplos")
print(f"Datos de validación: {len(validation_data[0])} ejemplos")
print(f"Datos de prueba: {len(test_data[0])} ejemplos")

# 3. Visualizar un ejemplo
idx = 0  # Puedes cambiar este índice para ver otras imágenes
imagen = training_data[0][idx]
etiqueta = training_data[1][idx]

print(f"Ejemplo de imagen (índice {idx}): Etiqueta = {etiqueta}")

# Convertir la imagen a 2D para visualizarla
imagen_2d = np.reshape(imagen, (28, 28))
plt.imshow(imagen_2d, cmap='gray')
plt.title(f"Etiqueta: {etiqueta}")
plt.show()

# Si ves la imagen y la información, ¡el primer paso está completo!
