"""
Archivo: nn_ejercicios/04_entrenar_red.py
Objetivo: Entrenar la red neuronal con los datos preparados y evaluar su rendimiento.

Instrucciones:
1. Crea una instancia de tu clase Network.
2. Entrena la red usando el método SGD con los datos preparados.
3. Evalúa el rendimiento en los datos de prueba y muestra el número de aciertos.

Tips:
- Usa los datos de 03_preparar_datos.py.
- Puedes usar epochs=10, mini_batch_size=10, eta=3.0 como valores iniciales.
- El método evaluate debe devolver el número de aciertos en test_data.

Comprobación automática:
"""
import numpy as np
import numpy.typing as npt
from typing import Final
from E02_red_neuronal import Network
from E01_cargar_datos import loadMnist

RawInput = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]

FILE_PATH: Final = 'input-files/mnist.pkl'
OUTPUT_LAYER_SIZE: Final = 10

def train_nn():
    training_data, validation_data, test_data = loadMnist(FILE_PATH)
    input_layer_size = len(training_data[0][0])
    net = Network([input_layer_size, 30, OUTPUT_LAYER_SIZE])

def test_train_nn():
    try:
        assert 'net' in globals(), "No se encontró la variable net (la red neuronal)."
        assert hasattr(net, 'SGD'), "La clase Network debe tener el método SGD."
        assert hasattr(net, 'evaluate'), "La clase Network debe tener el método evaluate."
        # Se espera que tras entrenar, la red tenga un accuracy > 80% en test_data_preparada
        accuracy = net.evaluate(test_data_preparada) / len(test_data_preparada)
        assert accuracy > 0.8, f"La red debe tener al menos 80% de aciertos en test_data_preparada. Accuracy actual: {accuracy:.2%}"
        print(f"¡Red entrenada correctamente! Accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"Error en la comprobación: {e}")
        raise
