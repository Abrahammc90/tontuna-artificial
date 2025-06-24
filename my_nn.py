import sys
import math
import numpy as np
import argparse
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

