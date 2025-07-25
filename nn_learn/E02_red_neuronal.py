"""
Archivo: nn_learn/02_red_neuronal.py
Objetivo: Implementar una red neuronal simple desde cero.

En este archivo crearás una clase Network que implementa una red neuronal multicapa (MLP) básica.

Pasos:
1. Definir la arquitectura de la red (número de capas y neuronas).
2. Inicializar los pesos y sesgos aleatoriamente.
3. Implementar la propagación hacia adelante (forward pass).
4. Implementar la función de coste (por ejemplo, error cuadrático medio).
5. Implementar el entrenamiento usando descenso de gradiente estocástico (SGD).
6. Probar la red con los datos de prueba.

Cada paso está explicado con comentarios detallados.
"""

import numpy as np
import random

class Network:
    def __init__(self, sizes):
        """
        sizes: lista con el número de neuronas en cada capa.
        Ejemplo: [784, 30, 10] para una red con 784 entradas, 1 capa oculta de 30 y 10 salidas.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Inicializa los sesgos y pesos con valores aleatorios
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        """Función de activación sigmoide."""
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        """Calcula la salida de la red para una entrada a."""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Entrena la red usando descenso de gradiente estocástico.
        training_data: lista de tuplas (x, y)
        epochs: número de épocas
        mini_batch_size: tamaño de los mini-batches
        eta: tasa de aprendizaje
        test_data: datos de prueba opcionales para evaluar el rendimiento
        """
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Época {j}: {self.evaluate(test_data)} / {len(test_data)} correctos")
            else:
                print(f"Época {j} completada")

    def update_mini_batch(self, mini_batch, eta):
        """
        Actualiza los pesos y sesgos usando un mini-batch y retropropagación.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Retropropagación: calcula el gradiente de la función de coste respecto a pesos y sesgos.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Propagación hacia adelante
        activation = x
        activations = [x] # lista para guardar todas las activaciones capa por capa
        zs = [] # lista para guardar todos los vectores z capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # Propagación hacia atrás
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        print(delta)
        exit()
        print(self.cost_derivative(activations[-1], y).shape)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        print(delta.shape, activations[-2].shape, "shape")
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        exit()
        return (nabla_b, nabla_w)

    def sigmoid_prime(self, z):
        """Derivada de la función sigmoide."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def evaluate(self, test_data):
        """
        Evalúa la red en los datos de prueba y devuelve el número de aciertos.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Derivada de la función de coste respecto a la salida.
        """
        return (output_activations - y)

# Para comprobar que la clase funciona, puedes crear una red y pasarle datos aleatorios:
if __name__ == "__main__":
    net = Network([784, 30, 10])
    x = np.random.randn(784, 1)
    salida = net.feedforward(x)
    print("Salida de la red para una entrada aleatoria:", salida)

# En el siguiente archivo aprenderás a preparar los datos para entrenar la red.
