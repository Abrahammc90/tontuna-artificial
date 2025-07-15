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
        self.a = activation_functions.ReLU(self.z)  # output
        return


class neural_network:

    def __init__(self, X: list[npt.NDArray[np.float64]], Y: list[npt.NDArray[np.float64]], hidden_layers_sizes):
        """Initialize the neural network with the given training inputs"""

        #add the output layer to the neuron numbers
        self.x = X[0]
        self.y = Y[0]
        input_layer_size = len(self.x)
        output_layer_size = len(self.y)
        self.layer_sizes = [input_layer_size] + hidden_layers_sizes + [output_layer_size]
        self.total_layers = len(self.layer_sizes)
        self.W = [np.random.rand(self.layer_sizes[i], self.layer_sizes[i-1]) for i in range(1, self.total_layers)]
        self.B = [np.random.rand(self.layer_sizes[i]) for i in range(1, self.total_layers)]
        self.Z = [np.zeros(self.layer_sizes[i]) for i in range(1, self.total_layers)]
        self.A = [self.x]
        self.A += [np.zeros(self.layer_sizes[i]) for i in range(1, self.total_layers)]

        return

    def feedforward(self):

        for i in range(1, self.total_layers):
            
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            self.A[i] = np.maximum(0, self.Z) #ReLU a todos los valores de Z
        
        self.C = (self.A - self.y)**2

        return


    def backpropagation(self):
        
        # x = self.X[0]
        # y = self.Y[0]
        gradiente = []

        DELTA_W = [np.array(np.zeros(self.layer_sizes[i], self.layer_sizes[i-1]) for i in range(1, self.total_layers))]
        DELTA_B = [np.array(np.zeros(self.layer_sizes[i]) for i in range(1, self.total_layers))]
        DELTA_A = [np.array(np.zeros(self.layer_sizes[i], self.layer_sizes[i-1]) for i in range(1, self.total_layers))]

        #Last layer
        aL_1 = self.A[-1]
        dCo_daL_1 = 2*(aL_1-self.y)
        
        zL_1 = self.Z[-1]
        daL_1_dzL_1 = (zL_1 > 0).astype(float)


        last_layer_error = daL_1_dzL_1 * dCo_daL_1

        aL_2 = self.A[-2]
        dzL_1_dw_1 = aL_2
        dzL_1_db_1 = 1

        dCo_db_1 = last_layer_error * dzL_1_db_1
        dCo_dw_1 = np.dot(last_layer_error, dzL_1_dw_1)


        #Last layer -1

        
        zL_2 = self.Z[-2]

        dzL_2_dw_2 = self.A[-3]
        dzl_2_db_2 = 1
        daL_2_dzl_2 = (zL_2 > 0).astype(float)

        
        dzL_1_aL_2 = self.W[-1]

        layer_error = np.dot(last_layer_error, dzL_1_aL_2)
        layer_error = np.dot(layer_error, daL_2_dzl_2)

        dCo_db_2 = layer_error * dzl_2_db_2
        dCo_dw_2 = np.dot(layer_error, dzL_2_dw_2)

        


        print(last_layer_error)
        exit()


        dCo_dleft_a = 0
        dCo_dw_2 = 0
        dCo_db_1 = 0



        for right_neuron in self.output_layer:
            for weight, left_neuron in zip(right_neuron.weights, self.hidden_layers[-1]): # pesos de la relación de la neurona derecha con todas las de la izquierda
                #dCo_dprev_a += np.gradient(z, prev_a) * np.gradient(a, z) * np.gradient(Co, a)
                left_a = left_neuron.a
                dCo_dleft_a += weight * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                dCo_dw_2 += left_a * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                dCo_db_1 += 1 * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)










        for i in range(len(self.hidden_layers)-1, 0, -1): # capas (output_layer)
            for right_neuron in self.hidden_layers[i]: # neuronas (primera output_neuron)
                dCo_dleft_a = 0
                dCo_dw_2 = 0
                dCo_db_1 = 0
                for weight, left_neuron in zip(right_neuron.weights, self.hidden_layers[i-1]): # pesos de la relación de la neurona derecha con todas las de la izquierda
                    #dCo_dprev_a += np.gradient(z, prev_a) * np.gradient(a, z) * np.gradient(Co, a)
                    left_a = left_neuron.a
                    dCo_dleft_a += weight * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                    dCo_dw_2 += left_a * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                    dCo_db_1 += 1 * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                gradiente += [dCo_dleft_a, dCo_dw_2, dCo_db_1]
                #dCo_db = np.gradient(z, b) * np.gradient(a, z) * np.gradient(Co, a)
        
        for right_neuron in self.hidden_layers[0]:
            dCo_dleft_a = 0
            dCo_dw_2 = 0
            dCo_db_1 = 0
            for weight, pixel in zip(right_neuron.weights, x): # pesos de la relación de la neurona derecha con todas las de la izquierda
                #dCo_dprev_a += np.gradient(z, prev_a) * np.gradient(a, z) * np.gradient(Co, a)
                left_a = pixel
                dCo_dleft_a += weight * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                dCo_dw_2 += left_a * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
                dCo_db_1 += 1 * activation_functions.ReLU_prima(left_a) * 2*(right_neuron.a - y)
            gradiente += [dCo_dleft_a, dCo_dw_2, dCo_db_1]

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

    nn = neural_network(training_x, training_y, [30, 30])
    nn.backpropagation()
    #nn.train()
    #print(nn.layers)

    
    
    #self.W.append(W0)

if __name__ == '__main__':
    main()

