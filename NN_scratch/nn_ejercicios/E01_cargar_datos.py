"""
Archivo: nn_ejercicios/01_cargar_datos.py
Objetivo: Cargar y explorar los datos de entrada para la red neuronal desde input-files.

Instrucciones:
1. Busca el archivo MNIST (input-files/mnist.pkl o mnist.pkl.gz).
2. Carga los datos de entrenamiento, validación y prueba usando pickle (y gzip si es necesario).
3. Muestra la cantidad de ejemplos en cada conjunto.
4. Visualiza una imagen de ejemplo y su etiqueta (opcional, recomendado).

Tips:
- Usa os.path para construir rutas.
- Usa pickle.load con encoding='latin1' si da error de decodificación.
- Para visualizar una imagen, usa matplotlib (plt.imshow) y reshape a (28, 28).

Al final, deja las siguientes variables:
- training_data, validation_data, test_data

Comprobación automática:
"""
import pickle

import numpy as np
import numpy.typing as npt

RawInput = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]

def loadMnist(filePath: str) -> tuple[RawInput, RawInput, RawInput]:
    """ Carga datos de archivo MNIST cuyo path se pasa por parámetro """
    with open(filePath, 'rb') as file:
        training_data, validation_data, test_data = pickle.load(file, encoding='latin1')
    print('-----------')
    print(f'training_data: [{type(training_data[0])}{training_data[0].shape}, {type(training_data[1])}{training_data[1].shape}]')
    print(f'\t training_data[0] (inputs): {type(training_data[0])} >> {training_data[0].shape}')
    print(f'\t training_data[1] (expected results): {type(training_data[1])} >> {training_data[1].shape}')
    print('-----------')
    print(f'validation_data: [{type(validation_data[0])}{validation_data[0].shape}, {type(validation_data[1])}{validation_data[1].shape}]')
    print(f'\t validation_data[0] (inputs): {type(validation_data[0])} >> {validation_data[0].shape}')
    print(f'\t validation_data[1] (expected results): {type(validation_data[1])} >> {validation_data[1].shape}')
    print('-----------')
    print(f'test_data: [{type(test_data[0])}{test_data[0].shape}, {type(test_data[1])}{test_data[1].shape}]')
    print(f'\t test_data[0] (inputs): {type(test_data[0])} >> {test_data[0].shape}')
    print(f'\t test_data[1] (expected results): {type(test_data[1])} >> {test_data[1].shape}')
    print('-----------')
    return training_data, validation_data, test_data

def test_loadMnist() -> None:
    # Comprobación automática de la función
    if __name__ == "__main__":
        # Cambia la ruta si tu archivo está en otro lugar
        file_path = 'input-files/mnist.pkl'
        try:
            training_data, validation_data, test_data = loadMnist(file_path)
            assert len(training_data[0]) == 50000, "El conjunto de entrenamiento debe tener 50000 ejemplos."
            assert len(validation_data[0]) == 10000, "El conjunto de validación debe tener 10000 ejemplos."
            assert len(test_data[0]) == 10000, "El conjunto de prueba debe tener 10000 ejemplos."
            print("¡Carga de datos correcta!")
        except Exception as e:
            print(f"Error en la comprobación: {e}")
            raise

if __name__ == "__main__":
    print('TESTING FUNCTION: loadMnist')
    test_loadMnist()