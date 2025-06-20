import sys
import math
import numpy as np
import argparse
#import simpy as sp

class neuron:

    def __init__(self):
        self.prev_a = 0
        self.w = 0
        self.b = 0


class layer:
     
    def __init__(self, neurons_number):
        self.layer = []
        self.neurons_number = neurons_number

    #def construct(self, neurons_number):
    #    self.neurons_number = neurons_number

class neural_network:


    def __init__(self):

        self.input_layer = []
        self.layers = []
        self.layers_number = 0
        self.array_neuron_numbers = []

    def construct(self, input_layer, array_neuron_numbers):

        self.layers = [layer()]*(len(array_neuron_numbers)+1)
        #test_array = [0]*len(array_neuron_numbers)
        
        #print(self.layers)

        for i in range(len(array_neuron_numbers)):
            self.layers[i+1].construct(array_neuron_numbers[i])

        #for i in range(len(array_neuron_numbers)):
        #    self.layers.append(layer(array_neuron_numbers[i]))

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
    arguments1 = arguments()
    arguments2 = arguments()
    print(arguments.wheels, arguments1.wheels, arguments2.wheels)
    arguments1.wheels = 2
    print(arguments.wheels, arguments1.wheels, arguments2.wheels)
    arguments2.wheels = 3
    print(arguments.wheels, arguments1.wheels, arguments2.wheels)

    #print(arguments.__dict__)
    exit()

    args.files
    args.num

    print(sys.argv[1:])
    exit()

    input_files = sys.argv[1:]


    nn = neural_network()
    nn.construct([0, 0, 0], [2, 3, 4])

if __name__ == '__main__':
    main()

