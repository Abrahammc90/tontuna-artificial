"""
Archivo: nn_ejercicios/03_preparar_datos.py
Objetivo: Preparar los datos para entrenar la red neuronal.

Instrucciones:
1. Convierte cada imagen de los datos de entrenamiento, validación y prueba en un vector columna (shape (784, 1)).
2. Convierte las etiquetas de entrenamiento a vectores one-hot de 10 dimensiones.
3. Crea listas de tuplas (x, y) para cada conjunto.

Tips:
- Usa numpy.reshape para cambiar la forma de los arrays.
- El vector one-hot debe tener un 1.0 en la posición de la etiqueta y 0.0 en el resto.

Variables esperadas al final:
- training_data_prepared, validation_data_prepared, test_data_prepared

Comprobación automática:
"""
import numpy as np
import numpy.typing as npt
from nn_ejercicios.E01_cargar_datos import loadMnist

RawInput = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
ReshapedInput = list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
OneHot = npt.NDArray[np.float64]

def _get_one_hot(digit: int, length: int) -> OneHot:
    oneHot = np.zeros((length, 1))
    oneHot[digit, 0] = 1
    # print(f'oneHot for digit: {digit} and length: {length} >> {", ".join(str(int(x)) for x in oneHot.flatten())}')
    return oneHot

def _reshapeInput(inputTuple: RawInput, input_length: int, output_length: int) -> ReshapedInput:
    return [
        (np.reshape(input_data, (input_length, 1)), _get_one_hot(label, output_length))
        for input_data, label in zip(inputTuple[0], inputTuple[1])
    ]

def prepareInputs() -> tuple[ReshapedInput, ReshapedInput, ReshapedInput]:
    training_data, validation_data, test_data = loadMnist('input-files/mnist.pkl')

    reshaped_training_data = _reshapeInput(training_data, 784, 10)
    reshaped_validation_data = _reshapeInput(validation_data, 784, 10)
    reshaped_test_data = _reshapeInput(test_data, 784, 10)

    return reshaped_training_data, reshaped_validation_data, reshaped_test_data
    


def test_prepareInputs():
    # Llama a la función que prepara los datos
    training_data_prepared, validation_data_prepared, test_data_prepared = prepareInputs()

    # Comprobaciones automáticas
    assert isinstance(training_data_prepared, list), "training_data_prepared debe ser una lista."
    assert isinstance(validation_data_prepared, list), "validation_data_prepared debe ser una lista."
    assert isinstance(test_data_prepared, list), "test_data_prepared debe ser una lista."

    x, y = training_data_prepared[0]
    assert x.shape == (784, 1), "Cada imagen debe ser un vector columna de 784x1."
    assert y.shape == (10, 1), "Cada etiqueta debe ser un vector one-hot de 10x1."
    print("¡Datos preparados correctamente!")

if __name__ == "__main__":
    print('TESTING FUNCTION: prepareInputs')
    test_prepareInputs()
