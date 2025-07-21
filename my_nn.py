import sys
import math
import numpy as np
import numpy.typing as npt
import argparse
import pickle
#import gzip
#import simpy as sp


class activation_functions:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(x):
        s = activation_functions.sigmoid(x)
        return s * (1 - s)
    
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

    def __init__(self, input_layer_size: int, output_layer_size: int, hidden_layers_sizes):
        """Initialize the neural network with the given training inputs"""

        #add the output layer to the neuron numbers
        # self.x = X[0]
        # self.y = Y[0]
        # input_layer_size = len(self.x)
        # output_layer_size = len(self.y)
        self.layer_sizes = hidden_layers_sizes + [output_layer_size]
        self.total_layers = len(self.layer_sizes)
        self.W = [np.random.rand(self.layer_sizes[0], input_layer_size)]
        self.W += [
            np.random.rand(self.layer_sizes[i], self.layer_sizes[i-1])
              for i in range(1, self.total_layers)
        ]
        self.B = [np.random.rand(self.layer_sizes[i]) for i in range(self.total_layers)]
        self.Z = [np.zeros(self.layer_sizes[i]) for i in range(self.total_layers)]
        # self.A = [self.x]
        self.A = [np.zeros(self.layer_sizes[i]) for i in range(self.total_layers)]

        return

    def feedforward(self, x, y):

        self.Z[0] = np.dot(self.W[0], x) + self.B[0]
        self.A[0] = np.maximum(0, self.Z[0])

        for i in range(1, self.total_layers-1):
            
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            self.A[i] = np.maximum(0, self.Z[i]) #ReLU a todos los valores de Z
        
        self.Z[-1] = np.dot(self.W[-1], self.A[-2]) + self.B[-1]
        self.A[-1] = activation_functions.sigmoid(self.Z[-1])

        self.C = (self.A[-1] - y)**2
        

        return


    def backpropagation(self, x, y):
        
        # x = self.X[0]
        # y = self.Y[0]
        self.gradient_w = [np.zeros((self.layer_sizes[0], len(x)))]
        self.gradient_w += [np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])) for i in range(1, self.total_layers)]
        self.gradient_b = [np.zeros((self.layer_sizes[i])) for i in range(self.total_layers)]

        #Last layer
        aL = self.A[-1]
        dCo_daL = 2*(aL-y)
        
        zL = self.Z[-1]
        daL_dzL = activation_functions.sigmoid(zL)


        layer_error = daL_dzL * dCo_daL

        dzL_db = 1
        aL_left = self.A[-2]
        dzL_dw = np.tile(aL_left, (len(self.A[-1]), 1))
        dCo_db = layer_error * dzL_db
        dCo_dw = np.dot(layer_error, dzL_dw)
        
        self.gradient_b[-1] = dCo_db
        self.gradient_w[-1] = dCo_dw
        
        for i in range(len(self.layer_sizes)-2, -1, -1):
            
            zL = self.Z[i]
            # Cuando i = 0 recoge el valor de píxeles de la capa input. Else, recoge las a de la capa izquierda.
            aL_left = [ x if(i==0) else self.A[i-1]]

            dzL_dw = np.tile(aL_left, (len(self.A[i]), len(aL_left)))
            dzL_db = 1
            daL_dzL = (zL > 0).astype(float)

            dzL_right_daL = self.W[i+1]

            layer_error = np.dot(layer_error, dzL_right_daL)
            layer_error = layer_error * daL_dzL

            dCo_db = layer_error * dzL_db
            dCo_dw = np.dot(layer_error, dzL_dw)

            self.gradient_b[i] = dCo_db
            self.gradient_w[i] = dCo_dw
        



    def learn(self, learn_rate):
        for i in range(1, len(self.layer_sizes)):
            self.W[i] = self.W[i] - learn_rate*self.gradient_w[i]
            self.B[i] = self.B[i] - learn_rate*self.gradient_b[i]

    def train(self, X, Y, learn_rate):

        #x = X[0]
        #y = Y[0]
        for x, y in zip(X, Y):
        #for i in range(1000):
            self.feedforward(x, y)
            self.backpropagation(x, y)
            self.learn(learn_rate)

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
    learn_rate = float(sys.argv[2])
    training_x, training_y = MNIST_loader.load_file(pkl_file)[:]


    input_layer_size = len(training_x[0])
    output_layer_size = len(training_y[0])
    nn = neural_network(input_layer_size, output_layer_size, [30, 30])
    nn.train(training_x, training_y, learn_rate)
    print('toecho')
    #nn.train()
    #print(nn.layers)

    
    
    #self.W.append(W0)

if __name__ == '__main__':
    main()

