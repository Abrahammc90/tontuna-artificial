    
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
        # ...existing code...

        #add the output layer to the neuron numbers
        # self.x = X[0]
        # self.y = Y[0]
        # input_layer_size = len(self.x)
        # output_layer_size = len(self.y)
        self.layer_sizes = hidden_layers_sizes + [output_layer_size]
        self.total_layers = len(self.layer_sizes)
        self.W = [np.random.randn(self.layer_sizes[0], input_layer_size)]
        self.W += [
            np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1])
              for i in range(1, self.total_layers)
        ]
        self.B = [np.random.randn(self.layer_sizes[i], 1) for i in range(self.total_layers)]
        self.Z = [np.zeros(self.layer_sizes[i]) for i in range(self.total_layers)]
        # self.A = [self.x]
        self.A = [np.zeros(self.layer_sizes[i]) for i in range(self.total_layers)]
        return

    def feedforward(self, x, y):

        self.Z[0] = np.dot(self.W[0], x) + self.B[0]
        #self.A[0] = np.maximum(0, self.Z[0])
        self.A[0] = activation_functions.sigmoid(self.Z[0])

        for i in range(1, self.total_layers-1):
            
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            #self.A[i] = np.maximum(0, self.Z[i]) #ReLU a todos los valores de Z
            self.A[i] = activation_functions.sigmoid(self.Z[i])
            
        
        self.Z[-1] = np.dot(self.W[-1], self.A[-2]) + self.B[-1]
        self.A[-1] = activation_functions.sigmoid(self.Z[-1])

        self.C = (self.A[-1] - y)**2
        
        
        if np.argmax(self.A[-1]) == np.argmax(y):
            return 1
        else:
            return 0

    
    def backpropagation(self, x, y):
        
        # x = self.X[0]
        # y = self.Y[0]
        
        # self.gradient_w = [np.zeros((self.layer_sizes[0], len(x)))]
        # self.gradient_w += [np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])) for i in range(1, self.total_layers)]
        # self.gradient_b = [np.zeros(((self.layer_sizes[i]), 1)) for i in range(self.total_layers)]
        
        #Last layer
        aL = self.A[-1]
        dCo_daL = 2*(aL-y)
        
        zL = self.Z[-1]
        daL_dzL = activation_functions.sigmoid_prime(zL)

        layer_error = daL_dzL * dCo_daL

        dzL_db = 1
        aL_left = self.A[-2]
        dzL_dw = np.tile(aL_left, (len(self.A[-1]), 1))
        dCo_db = layer_error * dzL_db

        dCo_dw = np.dot(layer_error, aL_left.transpose())
        
        # print(self.gradient_b[0].shape)
        # print(dCo_db.shape)
        #exit()

        self.gradient_b[-1] += dCo_db
        self.gradient_w[-1] += dCo_dw
        
        for i in range(len(self.layer_sizes)-2, -1, -1):
            
            zL = self.Z[i]
            # Cuando i = 0 recoge el valor de píxeles de la capa input. Else, recoge las a de la capa izquierda.
            aL_left = x if (i == 0) else self.A[i-1]

            dzL_dw = np.tile(aL_left, (len(self.A[i]), len(aL_left)))
            
            dzL_db = 1
            daL_dzL = activation_functions.sigmoid_prime(zL)

            dzL_right_daL = self.W[i+1]

            layer_error = np.dot(dzL_right_daL.transpose(), layer_error)
            layer_error = layer_error * daL_dzL

            dCo_db = layer_error * dzL_db
            dCo_dw = np.dot(layer_error, aL_left.transpose())

            self.gradient_b[i] += dCo_db
            self.gradient_w[i] += dCo_dw
      
    def learn(self, learn_rate):
        for i in range(len(self.layer_sizes)):
            self.W[i] = self.W[i] - learn_rate*self.gradient_w[i]
            self.B[i] = self.B[i] - learn_rate*self.gradient_b[i]

    def evaluate(self, test_x, test_y):
        """
        Evalúa el número de aciertos en test_x y test_y.
        """
        correct = 0
        for x, y in zip(test_x, test_y):
            output = self.feedforward(x, y)
            correct += output
        return correct, len(test_x)


    def train(self, X, Y, learn_rate, epochs=10, mini_batch_size=32, test_x=None, test_y=None):
        data_size = len(X)
        for epoch in range(epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            Y_shuffled = [Y[i] for i in indices]
            mini_batches = [
                (X_shuffled[k:k+mini_batch_size], Y_shuffled[k:k+mini_batch_size])
                for k in range(0, data_size, mini_batch_size)
            ]
            total_correct = 0
            total_samples = 0
            for X_batch, Y_batch in mini_batches:
                self.gradient_w = [np.zeros((self.layer_sizes[0], len(X[0])))]
                self.gradient_w += [np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])) for i in range(1, self.total_layers)]
                self.gradient_b = [np.zeros(((self.layer_sizes[i]), 1)) for i in range(self.total_layers)]
                local_correct = 0
                local_total_samples = 0
                for x, y in zip(X_batch, Y_batch):
                    local_correct += self.feedforward(x, y)
                    self.backpropagation(x, y)
                self.learn(learn_rate/len(X_batch))
                total_correct += local_correct
                local_total_samples += len(X_batch)
                total_samples += len(X_batch)
                print(f"Epoch {epoch+1} - Mini-batch: {total_samples}/{data_size} - Aciertos: {local_correct}/{local_total_samples} ({(local_correct/local_total_samples)*100:.2f}%)")
                #if test_x is not None and test_y is not None:
                #    test_correct, test_total = self.evaluate(test_x, test_y)
                #    print(f"Test: {test_correct}/{test_total} aciertos ({(test_correct/test_total)*100:.2f}%)")
            print(f"Epoch {epoch+1} completada. Aciertos totales: {total_correct}/{data_size} ({(total_correct/data_size)*100:.2f}%)")

    

class MNIST_loader:

    @staticmethod
    def load_file(dataFile):
        f = open(dataFile, 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
        f.close()
        training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
        training_results = [MNIST_loader.vectorize_result(y) for y in training_data[1]]
        training_dataset = [training_inputs, training_results]
        return [training_dataset, test_data]

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

    pkl_file = sys.argv[1]
    learn_rate = float(sys.argv[2])
    training, test = MNIST_loader.load_file(pkl_file)[:]
    training_x, training_y = training[:]
    test_x, test_y = test[:]


    input_layer_size = len(training_x[0])
    output_layer_size = len(training_y[0])
    nn = neural_network(input_layer_size, output_layer_size, [30, 30])
    nn.train(training_x, training_y, learn_rate, test_x=test_x, test_y=test_x)
    print('toecho')
    #nn.train()
    #print(nn.layers)

    
    
    #self.W.append(W0)

if __name__ == '__main__':
    main()

