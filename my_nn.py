import sys
import math
import numpy as np
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

class neuron:

    def __init__(self, activation_function):
        self.w = np.random.rand()  # weight
        self.b = np.random.rand()  # bias
        self.activation_function = activation_function
        return

    def update(self, prev_a):
        #sum_prev_a = np.sum(np.array(prev_a))
        z = self.w*prev_a + self.b  # activation
        #sum_z = 0 # sum over all previous neurons
        self.a = self.activation_function(z)  # output
        return


class neural_network:

    def __init__(self, training_input_data, len_hidden_layers, len_output_layer):
        """Initialize the neural network with the given training inputs"""

        #add the output layer to the neuron numbers
        total_neuron_layers = [len(training_input_data)] + len_hidden_layers + [len_output_layer]
        #self.input_layer = [neuron()]*len_input_layer
        #self.output_layer = len_output_layer
        #activation_functions = [ReLU]*len(total_neuron_layers)-1 + [sigmoid]
        last_layer_index = len(total_neuron_layers)-1
        self.layers = [[neuron( activation_functions.sigmoid if n == last_layer_index else activation_functions.ReLU )]*n for n in total_neuron_layers]

        for pixel, input_neuron in zip(training_input_data[0], self.layers[0]):
            input_neuron.update(pixel)

        #self.layers = [[neuron(activation_functions[i])]*total_neuron_layers[i] for i in range(len(total_neuron_layers))]

        return

    def feedforward(self, X):
        
        x = X[0]

        for pixel, input_neuron in zip(x, self.layers[0]):
            input_neuron.update(pixel)

        for i in range(len(self.layers[1:])):
            sum_a = np.sum(np.array([prev_neuron.a for prev_neuron in self.layers[i]]))
            for next_neuron in self.layers[i+1]:
                next_neuron.update(sum_a)



        
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
        training_dataset = [training_data[0], training_results]
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

