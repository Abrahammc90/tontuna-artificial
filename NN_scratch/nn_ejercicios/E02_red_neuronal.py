"""
Archivo: nn_ejercicios/02_red_neuronal.py
Objetivo: Implementar una red neuronal simple desde cero.

Instrucciones:
1. Crea una clase Network que reciba una lista con el número de neuronas por capa.
2. Inicializa los pesos y sesgos aleatoriamente.
3. Implementa la función de activación sigmoide.
4. Implementa el método feedforward para calcular la salida de la red.

Tips:
- Usa numpy para los arrays y operaciones.
- La arquitectura típica para MNIST es [784, 30, 10].
- feedforward debe aceptar un vector columna (shape (784, 1)).

Al final, deja la clase Network implementada y crea una instancia con la arquitectura anterior.

Comprobación automática:
"""
import numpy as np
import numpy.typing as npt

class Network:
    biases: list[npt.NDArray[np.float64]]
    weights: list[npt.NDArray[np.float64]]

    def __init__(self, neurons_per_layer: list[int]):
        # Biases: one vector per layer except the input layer
        self.biases = [
            np.random.randn(y, 1)
            for y in neurons_per_layer[1:]
        ]

        # Weights: one matrix per connection between layers
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(neurons_per_layer[:-1], neurons_per_layer[1:])
        ]

    @staticmethod
    def sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-z))
    
    def feedforward(self, input: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calcula la salida de la red neuronal para una entrada dada usando propagación hacia adelante.
        input: vector columna (n, 1) con la entrada a la red.
        Devuelve: vector columna con la salida de la red.
        """
        # La variable 'a' representa la activación actual (la entrada a la red en la primera capa)
        activations_from_previous_layer = input
        
        # prerellenamos el resultado con ceros para que no se pueda devolver vacío
        # Sacamos la longitud del primer elemento de los pesos porque es un array de 3 dimensiones:
        # - 1: conexiones entre capas
        # - - 1: empezamos entre la primera capa (input) y la segunda, y tenemos 
        # - - - 1: respuestas a preparar para la siguiente capa
        # - - - 2: los pesos a aplicar a cada valor que nos llega de la capa anterior 
        
        # Recorremos cada par de pesos y sesgos de cada capa de la red (excepto la de entrada)
        # ⚠ Ojo, este bucle recorre TODAS las capas y calcula las respuestas finales de toda la red
        for weights_current_layer, biases_current_layer in zip(self.weights, self.biases):
            # Calculamos la activación de esta capa: w(L) @ a(L-1) + b(L)
            activations_current_layer =np.dot(weights_current_layer, activations_from_previous_layer) + biases_current_layer
            # Preparamos la siguiente vuelta del for que es el cálculo de la activación de la capa siguiente
            activations_from_previous_layer = self.sigmoid(activations_current_layer)
        
        return activations_from_previous_layer

def test_Network() -> None:
    try:
        assert 'Network' in globals(), "No se encontró la clase Network."
        net = Network([784, 30, 10])
        print(f'[no input] net.biases: [{type(net.biases[0])}{net.biases[0].shape}, {type(net.biases[1])}{net.biases[1].shape}]')
        print(f'[no input] net.weights: [{type(net.weights[0])}{net.weights[0].shape}, {type(net.weights[1])}{net.weights[1].shape}]')
        x = np.random.randn(784, 1)
        print(f'x (input): {type(x)}{x.shape}')
        salida = net.feedforward(x)
        print(f'net.feedforward(x): {type(salida)}{salida.shape}')
        assert salida.shape == (10, 1), "La salida de la red debe ser un vector columna de 10x1."
        print("¡Red neuronal creada y feedforward funciona!")
    except Exception as e:
        print(f"Error en la comprobación: {e}")
        raise

if __name__ == "__main__":
    print('TESTING CLASS: Network')
    test_Network()
