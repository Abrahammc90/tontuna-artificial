import sys
import math
import numpy as np
import argparse
import cPickle
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

    def __init__(self, input_layer, array_neuron_numbers):

        self.input_layer = input_layer
        self.layers = [[neuron()]*n for n in array_neuron_numbers]

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
     
    def __init__(self, datafile):
        f = open(datafile, 'rb')
        self.training_data, self.validation_data, self.test_data = cPickle.load(f)
        f.close()

    def wrap(self):
        training_inputs = [np.reshape(x, (784, 1)) for x in self.training_data[0]]
        training_results = [self.vectorized_result(y) for y in self.training_data[1]]
        training_data = zip(training_inputs, training_results)
        validation_inputs = [np.reshape(x, (784, 1)) for x in self.validation_data[0]]
        validation_data = zip(validation_inputs, self.validation_data[1])
        test_inputs = [np.reshape(x, (784, 1)) for x in self.test_data[0]]
        test_data = zip(test_inputs, self.test_data[1])
        return (training_data, validation_data, test_data)
    
    def vectorized_result(j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
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

    #args = arguments()
    #args.parse(sys.argv)
    #args = arguments()
    #arguments1 = arguments()
    #arguments2 = arguments()
    #print(arguments.wheels, arguments1.wheels, arguments2.wheels)
    #arguments1.wheels = 2
    #print(arguments.wheels, arguments1.wheels, arguments2.wheels)
    #arguments2.wheels = 3
    #print(arguments.wheels, arguments1.wheels, arguments2.wheels)

    #print(arguments.__dict__)
    #exit()

    #input_files = sys.argv[1:]


    nn = neural_network([1,2,3,4,5,6,7,8], [4, 2, 3])
    #print(nn.layers)

if __name__ == '__main__':
    main()

