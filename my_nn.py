import sys
import math
import numpy as np
import argparse
import pickle
import gzip
#import simpy as sp

class neuron:

    def __init__(self):
        self.w = np.random.rand()  # weight
        self.b = np.random.rand()  # bias
        return

    def update(self, prev_a, activation_function):

        self.z = self.w*prev_a + self.b  # activation
        self.a = activation_function(self.z)  # output
        return


class neural_network:

    def __init__(self, training_inputs, expected_results, hidden_layer_neuron_numbers):
        """Initialize the neural network with the given training inputs"""

        #add the output layer to the neuron numbers
        total_neuron_layers = hidden_layer_neuron_numbers + [len(expected_results[0])]
        self.input_layer = training_inputs
        self.hidden_layers = [[neuron()]*n for n in total_neuron_layers]

        return

    #def backpropagation(self, layer_0, layer_1):
#
#
#
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


def sigmoid(x):
        fx = 1/(1+math.e**(-x))
        return fx

class MNIST_loader:

    def __init__(self, dataFile):
        f = open(dataFile, 'rb')
        self.training_data, self.validation_data, self.test_data = pickle.load(f, encoding="bytes")
        self.training_results = [self.vectorize_result(y) for y in self.training_data[1]]
        f.close()

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
    
    def vectorize_result(self, j):
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
    input_data = MNIST_loader(pkl_file)

    nn = neural_network(input_data.training_data[0], input_data.training_results, [30, 30])
    #print(nn.layers)

    
        

if __name__ == '__main__':
    main()

