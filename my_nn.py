import sys
import math
import numpy as np
import numpy.typing as npt
import argparse
import pickle
import gzip
#import simpy as sp


class activation_functions:

    @staticmethod
    def sigmoid(x):
        fx = 1/(1+math.e**(-x))
        return fx
    
    @staticmethod
    def ReLU(x):
        if x > 0:
            return x
        else:
            return 0

    @staticmethod
    def ReLU_prima(x):
        if x > 0:
            return 1
        else:
            return 0

class neuron:

    def __init__(self, weights_number):
        self.weights = [np.random.rand()]*weights_number  # weight
        self.b = np.random.rand()  # bias
        return

    def update(self, prev_activations):
        self.z = 0
        for w, prev_a in zip(self.weights, prev_activations):
            self.z += w*prev_a # activation
        self.z += self.b
        self.a = activation_functions.ReLU(z)  # output
        return


class neural_network:

    def __init__(self, len_input_layer, len_hidden_layers, len_output_layer):
        """Initialize the neural network with the given training inputs"""

        #add the output layer to the neuron numbers
        total_neuron_layers = [len_input_layer] + len_hidden_layers + [len_output_layer]
        #self.input_layer = [neuron()]*len_input_layer
        #self.output_layer = len_output_layer
        #activation_functions = [ReLU]*len(total_neuron_layers)-1 + [sigmoid]
        self.hidden_layers = [[neuron(total_neuron_layers[i-1])]*total_neuron_layers[i] for i in range(1, len(total_neuron_layers))]

        #self.layers = [[neuron(activation_functions[i])]*total_neuron_layers[i] for i in range(len(total_neuron_layers))]

        return

    def feedforward(self, X ):
        x = X[0]

        for neuron in self.hidden_layers[0]:
            neuron.update(x)

        for i in range(1, len(self.hidden_layers)):
            for right_neuron in self.hidden_layers[i]:
                right_neuron.update([left_neuron.a for left_neuron in self.hidden_layers[i-1]])

    def backpropagation(self, training_x, training_y):
        
        x = training_x[0]
        y = training_y[0]
        gradiente = []
        for i in range(len(self.hidden_layers)-1, 0, -1): # capas (output_layer)
            for right_neuron in self.hidden_layers[i]: # neuronas (primera output_neuron)
                dCo_dleft_a = 0
                dCo_dw = 0
                dCo_db = 0
                for weight, left_neuron in zip(right_neuron.weights, self.hidden_layers[i-1]): # pesos de la relación de la neurona derecha con todas las de la izquierda
                    #dCo_dprev_a += np.gradient(z, prev_a) * np.gradient(a, z) * np.gradient(Co, a)
                    left_a = left_neuron.a
                    dCo_dleft_a += weight * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                    dCo_dw += left_a * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                    dCo_db += 1 * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                gradiente += [dCo_dleft_a, dCo_dw, dCo_db]
                #dCo_db = np.gradient(z, b) * np.gradient(a, z) * np.gradient(Co, a)
        
        for right_neuron in self.hidden_layers[0]:
            dCo_dleft_a = 0
            dCo_dw = 0
            dCo_db = 0
            for weight, pixel in zip(right_neuron.weights, x): # pesos de la relación de la neurona derecha con todas las de la izquierda
                #dCo_dprev_a += np.gradient(z, prev_a) * np.gradient(a, z) * np.gradient(Co, a)
                left_a = pixel
                dCo_dleft_a += weight * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                dCo_dw += left_a * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                dCo_db += 1 * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
            gradiente += [dCo_dleft_a, dCo_dw, dCo_db]

            #   z = w*prev_a+b
            #   a = ReLU(z)
            #   Co = (a-y)^2
            #   
            #   dz_dprev_a = w
            #
            #   dCo_dw = np.gradient(z, w) * np.gradient(a, z) * np.gradient(Co, a)
            #   dCo_dprev_a = np.gradient(z, prev_a) * np.gradient(a, z) * np.gradient(Co, a)
            #   dCo_db = np.gradient(z, b) * np.gradient(a, z) * np.gradient(Co, a)
            #   return([dCo_dw, dCo_dprev_a, dCo_db])
            

"""
La fórmula de backpropagation en el descenso de gradiente para redes neuronales multicapa permite calcular cómo ajustar cada peso y sesgo para minimizar el error de la red.
Se basa en la regla de la cadena y el cálculo de derivadas parciales de la función de coste respecto a cada parámetro.

Fórmulas clave de backpropagation
Supón una red con función de coste ( C ), pesos ( w ), sesgos ( b ), activaciones ( a ), entradas ponderadas ( z ), y función de activación ( \sigma ).

1. Error en la capa de salida
[ \delta^L = \nabla_a C \odot \sigma'(z^L) ]

( \delta^L ): error en la capa de salida
( \nabla_a C ): derivada del coste respecto a la activación de salida
( \odot ): producto elemento a elemento
( \sigma'(z^L) ): derivada de la función de activación evaluada en la entrada ponderada de la capa de salida
2. Error en capas anteriores (propagación hacia atrás)
[ \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l) ]

( \delta^l ): error en la capa ( l )
( w^{l+1} ): matriz de pesos de la capa siguiente
( \delta^{l+1} ): error de la capa siguiente
( \sigma'(z^l) ): derivada de la activación en la capa actual
3. Gradientes para actualizar pesos y sesgos
[ \frac{\partial C}{\partial b^l_j} = \delta^l_j ] [ \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \cdot \delta^l_j ]

El gradiente respecto al sesgo es el error de la neurona.
El gradiente respecto al peso es la activación de la neurona anterior multiplicada por el error de la neurona actual.
Resumen del algoritmo
Feedforward: Calcula todas las activaciones y entradas ponderadas ( z ).
Backpropagation:
Calcula el error en la capa de salida (( \delta^L )).
Propaga el error hacia atrás usando la fórmula recursiva para ( \delta^l ).
Calcula los gradientes para cada peso y sesgo.
Actualiza pesos y sesgos:
Resta una fracción del gradiente (multiplicada por la tasa de aprendizaje) a cada peso y sesgo.
"""



        
#
#
        #for neuron_layer1:
        #for neuron_layer2:
        #for neuron_layer3:
                        
    

    #def backpropagation():

    #    z = w*prev_a+b
    #    a = sigmoid(z)
    #    Co = (a-y)^2
#
#
#
    #    dCo_dw = np.gradient(z, w) * np.gradient(a, z) * np.gradient(Co, a)
    #    dCo_dprev_a = np.gradient(z, prev_a) * np.gradient(a, z) * np.gradient(Co, a)
    #    dCo_db = np.gradient(z, b) * np.gradient(a, z) * np.gradient(Co, a)
#
    #    return([dCo_dw, dCo_dprev_a, dCo_db])
#
    #def gradient(self, neurons):
#
    #    grad = []
    #    for i in neurons:
    #        grad += self.backpropagation(neurons[i-1].a, neurons[i].w, neurons[i].b)




class MNIST_loader:

    @staticmethod
    def load_file(dataFile):
        f = open(dataFile, 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
        f.close()
        training_results = [MNIST_loader.vectorize_result(y) for y in training_data[1]]
        training_dataset = [np.array(training_data[0]), training_results]
        return training_dataset

    # def _wrap(self):

    #     print("Wrapping training data")
    #     self.training_inputs = [np.reshape(x, 784) for x in self.training_data[0]]
    #     self.training_results = [self.vectorize_result(y) for y in self.training_data[1]]
    #     #training_data = zip(training_inputs, training_results)

    #     print("Wrapping validation data")
    #     self.validation_inputs = [np.reshape(x, 784) for x in self.validation_data[0]]
    #     #validation_data = zip(validation_inputs, self.validation_data[1])
        
    #     print("Wrapping test data")
    #     self.test_inputs = [np.reshape(x, 784) for x in self.test_data[0]]
    #     #test_data = zip(test_inputs, self.test_data[1])
    #     return #(training_data, validation_data, test_data)
    
    @staticmethod
    def vectorize_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros(10)
        e[j] = 1.0
        return e

class arguments:
    wheels = 4
    def __init__(self):
        self.files = []
        self.neuron_numbers = []
    #def parse(self, args):
    #    #arguments_parsed = argparse()
    #    self.files = arguments_parsed.files
    #    self.neuron_numbers = arguments_parsed.files
        

def main():

    pkl_file = sys.argv[1]
    training_x, training_y = MNIST_loader.load_file(pkl_file)[:]

    nn = neural_network(len(training_x[0]), [30, 30], len(training_y[0]))
    nn.train(training_x, training_y)
    #print(nn.layers)

    
        

if __name__ == '__main__':
    main()

