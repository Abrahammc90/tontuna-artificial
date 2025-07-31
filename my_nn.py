    
import sys
import math
#import numpy as np
import numpy.typing as npt
import argparse
import pickle
import numpy as np
import random
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
    
    #@staticmethod
    #def ReLU(x):
    #    if x > 0:
    #        return x
    #    else:
    #        return 0
    #    
    #@staticmethod
    #def ReLU_prima(x):
    #    if x > 0:
    #        return 1
    #    else:
    #        return 0
        
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_prime(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)


class neural_network:
    def __init__(self, input_layer_size: int, output_layer_size: int, hidden_layers_sizes, activation_function, use_cupy=False):
        
        #add the output layer to the neuron numbers
        # self.x = X[0]
        # self.y = Y[0]
        # input_layer_size = len(self.x)
        # output_layer_size = len(self.y)
        self.layer_sizes = hidden_layers_sizes + [output_layer_size]
        self.total_layers = len(self.layer_sizes)
        if use_cupy:
            self.W = [cp.random.randn(self.layer_sizes[0], input_layer_size)]
            self.W += [
                cp.random.randn(self.layer_sizes[i], self.layer_sizes[i-1])
                  for i in range(1, self.total_layers)
            ]
            self.B = [cp.random.randn(self.layer_sizes[i], 1) for i in range(self.total_layers)]
            self.Z = [cp.zeros(self.layer_sizes[i]) for i in range(self.total_layers)]
            # self.A = [self.x]
            self.A = [cp.zeros(self.layer_sizes[i]) for i in range(self.total_layers)]
        else:
            
            if activation_function == "relu":
                self.W = [np.random.randn(self.layer_sizes[0], input_layer_size) * np.sqrt(2/input_layer_size)] # Inicialización He
                self.W += [
                    np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(2/self.layer_sizes[i-1]) # Inicialización He
                      for i in range(1, self.total_layers-1)
                ]
                self.W += [
                    np.random.randn(self.layer_sizes[-1], self.layer_sizes[-2])
                ]
                self.B = [np.zeros((self.layer_sizes[i], 1)) + 0.1 for i in range(self.total_layers)]
                #self.B = [np.random.randn(self.layer_sizes[i], 1) for i in range(self.total_layers)]
            elif activation_function == "sigmoid":
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

    def feedforward(self, x, y, activation_function='sigmoid', use_cupy=False, test=False):
        
        if use_cupy:
            self.Z[0] = cp.dot(self.W[0], x) + self.B[0]
            if activation_function == 'relu':
                #self.A[0] = cp.maximum(0, self.Z[0])
                self.A[0] = activation_functions.leaky_relu(self.Z[0]) #ReLU a todos los valores de Z
            elif activation_function == 'sigmoid':
                self.A[0] = activation_functions.sigmoid(self.Z[0])  # Sigmoid activation for the first layer

            for i in range(1, self.total_layers-1):

                self.Z[i] = cp.dot(self.W[i], self.A[i-1]) + self.B[i]
                if activation_function == 'relu':
                    #self.A[i] = np.maximum(0, self.Z[i]) #ReLU a todos los valores de Z
                    self.A[i] = activation_functions.leaky_relu(self.Z[i]) #ReLU a todos los valores de Z
                elif activation_function == 'sigmoid':
                    self.A[i] = activation_functions.sigmoid(self.Z[i])


            self.Z[-1] = cp.dot(self.W[-1], self.A[-2]) + self.B[-1]
            self.A[-1] = activation_functions.sigmoid(self.Z[-1])
        else:
            self.Z[0] = np.dot(self.W[0], x) + self.B[0]
            if activation_function == 'relu':
                self.A[0] = np.maximum(0, self.Z[0])
            elif activation_function == 'sigmoid':
                self.A[0] = activation_functions.sigmoid(self.Z[0])  # Sigmoid activation for the first layer
    
            for i in range(1, self.total_layers-1):
                
                self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
                if activation_function == 'relu':
                    self.A[i] = np.maximum(0, self.Z[i]) #ReLU a todos los valores de Z
                elif activation_function == 'sigmoid':
                    self.A[i] = activation_functions.sigmoid(self.Z[i])
                
            
            self.Z[-1] = np.dot(self.W[-1], self.A[-2]) + self.B[-1]
            self.A[-1] = activation_functions.sigmoid(self.Z[-1])

        #self.C = (self.A[-1] - y)**2
        #eps = 1e-10  # Small value to avoid log(0)
        #self.C = - (y * np.log(aL + eps) + (1 - y) * np.log(1 - aL + eps))
        if test == True:
            print(np.argmax(self.A[-1]), np.argmax(y))
        if np.argmax(self.A[-1]) == np.argmax(y):
            return 1
        else:
            return 0
    
    def backpropagation(self, x, y, activation_function, use_cupy=False):
        
        # x = self.X[0]
        # y = self.Y[0]
        
        # self.gradient_w = [np.zeros((self.layer_sizes[0], len(x)))]
        # self.gradient_w += [np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])) for i in range(1, self.total_layers)]
        # self.gradient_b = [np.zeros(((self.layer_sizes[i]), 1)) for i in range(self.total_layers)]
        
        #Last layer
        aL = self.A[-1]
        dCo_daL = (aL-y)
        
        zL = self.Z[-1]
        daL_dzL = activation_functions.sigmoid_prime(zL)

        layer_error = daL_dzL * dCo_daL
        #layer_error = dCo_daL

        dzL_db = 1
        aL_left = self.A[-2]
        #dzL_dw = np.tile(aL_left, (len(self.A[-1]), 1))
        dCo_db = layer_error * dzL_db

        if use_cupy:
            dCo_dw = cp.dot(layer_error, aL_left.transpose())
        else:
            dCo_dw = np.dot(layer_error, aL_left.transpose())
        
        #print(dCo_dw.shape)
        #print(dCo_db.shape)
        #print()
        # print(self.gradient_b[0].shape)
        # print(dCo_db.shape)
        #exit()

        # Ensure correct array addition using numpy/cupy add
        if use_cupy:
            self.gradient_b[-1] = cp.add(self.gradient_b[-1], cp.array(dCo_db))
            self.gradient_w[-1] = cp.add(self.gradient_w[-1], cp.array(dCo_dw))
        else:
            self.gradient_b[-1] = np.add(self.gradient_b[-1], dCo_db)
            self.gradient_w[-1] = np.add(self.gradient_w[-1], dCo_dw)
        
        for i in range(len(self.layer_sizes)-2, -1, -1):
            
            zL = self.Z[i]
            # Cuando i = 0 recoge el valor de píxeles de la capa input. Else, recoge las a de la capa izquierda.
            aL_left = x if (i == 0) else self.A[i-1]

            #dzL_dw = np.tile(aL_left, (len(self.A[i]), len(aL_left)))
            
            dzL_db = 1

            if activation_function == 'relu':
                #daL_dzL = (zL > 0).astype(float)
                daL_dzL = activation_functions.leaky_relu_prime(zL)
            elif activation_function == 'sigmoid':
                daL_dzL = activation_functions.sigmoid_prime(zL)

            dzL_right_daL = self.W[i+1]

            if use_cupy:
                layer_error = cp.dot(dzL_right_daL.transpose(), layer_error)
            else:
                layer_error = np.dot(dzL_right_daL.transpose(), layer_error)
            layer_error = layer_error * daL_dzL

            dCo_db = layer_error * dzL_db
            if use_cupy:
                dCo_dw = cp.dot(layer_error, aL_left.transpose())
            else:
                dCo_dw = np.dot(layer_error, aL_left.transpose())

            #print(dCo_dw.shape)
            #print(dCo_db.shape)
            #print()

            if use_cupy:
                self.gradient_b[i] = cp.add(self.gradient_b[i], cp.array(dCo_db))
                self.gradient_w[i] = cp.add(self.gradient_w[i], cp.array(dCo_dw))
            else:
                self.gradient_b[i] = np.add(self.gradient_b[i], dCo_db)
                self.gradient_w[i] = np.add(self.gradient_w[i], dCo_dw)
        #return (self.gradient_b, self.gradient_w)
      
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
            output = self.feedforward(x, y, test=True)
            correct += output
        return correct, len(test_x)

    def train(self, X, Y, activation_function, learn_rate, epochs=20, mini_batch_size=10, use_cupy=False):
        """
        Vectorized training loop: avoids per-sample Python loops and list.append by pre-initializing arrays.
        X, Y: lists of column vectors (shape (n_in,1) and (n_out,1))
        """
        xp = cp if use_cupy else np
        data_size = len(X)

        for epoch in range(epochs):
            # Shuffle indices
            idx = xp.arange(data_size)
            xp.random.shuffle(idx)
            # Reorder data via list indexing
            X_shuf = [X[i] for i in idx.tolist()]
            Y_shuf = [Y[i] for i in idx.tolist()]

            # Mini-batches
            for start in range(0, data_size, mini_batch_size):
                batch = X_shuf[start:start + mini_batch_size]
                labels = Y_shuf[start:start + mini_batch_size]
                # Stack into matrices: shape (n_features, batch_size)
                Xb = xp.hstack(batch)
                Yb = xp.hstack(labels)
                m = Xb.shape[1]

                # Initialize gradients
                self.gradient_w = [xp.zeros_like(w) for w in self.W]
                self.gradient_b = [xp.zeros_like(b) for b in self.B]

                # Pre-allocate activations and z arrays
                activations = [None] * (self.total_layers + 1)
                zs = [None] * self.total_layers
                activations[0] = Xb

                # Forward pass
                for l in range(self.total_layers):
                    z = xp.dot(self.W[l], activations[l]) + self.B[l]
                    zs[l] = z
                    if activation_function == 'relu' and l < self.total_layers -1:
                        a = xp.maximum(0, z)
                    else:
                        a = activation_functions.sigmoid(z)
                    activations[l + 1] = a

                # Backward pass: output layer error
                sp = activation_functions.sigmoid_prime(zs[-1])
                delta = (activations[-1] - Yb) * sp
                # Gradients for last layer
                self.gradient_b[-1] = xp.sum(delta, axis=1, keepdims=True)
                self.gradient_w[-1] = xp.dot(delta, activations[-2].T)

                # Backpropagate through remaining layers
                for l in range(self.total_layers -2, -1, -1):
                    if activation_function == 'relu':
                        sp = xp.where(zs[l] > 0, 1, 0)
                    else:
                        sp = activation_functions.sigmoid_prime(zs[l])
                    delta = xp.dot(self.W[l + 1].T, delta) * sp
                    self.gradient_b[l] = xp.sum(delta, axis=1, keepdims=True)
                    self.gradient_w[l] = xp.dot(delta, activations[l].T)

                # Update parameters
                for i in range(self.total_layers):
                    self.W[i] -= (learn_rate / m) * self.gradient_w[i]
                    self.B[i] -= (learn_rate / m) * self.gradient_b[i]

            # Optional: compute epoch accuracy
            preds = [self.feedforward(x,y,activation_function, use_cupy) for x,y in zip(X, Y)]
            acc = sum(preds) / data_size
            print(f"Epoch {epoch+1} — Accuracy: {acc*100:.2f}%")

    def tensor_train(self, X, Y, activation_function, learn_rate, epochs=20, mini_batch_size=10, use_cupy=False):
        """
        Vectorized mini-batch training via 3D tensors and tensordot for batched matmuls.
        Supports sigmoid and leaky ReLU activations.
        X, Y: lists of column vectors (shape (n_in,1) and (n_out,1)).
        """
        xp = cp if use_cupy else np
        alpha = 0.01  # Leaky ReLU negative slope
        # Stack dataset once
        X_np = np.hstack(X)
        Y_np = np.hstack(Y)
        if use_cupy:
            X_mat = cp.asarray(X_np, dtype=cp.float32)
            Y_mat = cp.asarray(Y_np, dtype=cp.float32)
            # Transfer parameters
            self.W = [cp.asarray(w, dtype=cp.float32) for w in self.W]
            self.B = [cp.asarray(b, dtype=cp.float32) for b in self.B]
        else:
            X_mat = X_np.astype(np.float32)
            Y_mat = Y_np.astype(np.float32)

        # Dimensions
        n_in, data_size = X_mat.shape
        n_out = Y_mat.shape[0]
        B = mini_batch_size
        N = (data_size + B - 1) // B  # number of batches

        for epoch in range(epochs):
            # Shuffle and possibly pad
            perm = xp.random.permutation(data_size)
            Xs = X_mat[:, perm]
            Ys = Y_mat[:, perm]
            pad = N * B - data_size
            if pad > 0:
                Xs = xp.pad(Xs, ((0, 0), (0, pad)))
                Ys = xp.pad(Ys, ((0, 0), (0, pad)))
            # Shape into batches: (features, B, N)
            Xb = Xs.reshape(n_in, B, N)
            Yb = Ys.reshape(n_out, B, N)

            # Pre-allocate gradient accumulators
            grads_w = [xp.zeros_like(w) for w in self.W]
            grads_b = [xp.zeros_like(b) for b in self.B]

            # Forward pass across all batches
            activ = [Xb]
            zs = []
            for l in range(self.total_layers):
                # Compute z with tensordot: sum over input features
                z = xp.tensordot(self.W[l], activ[l], axes=([1], [0])) + self.B[l][..., None, None]
                zs.append(z)
                # Activation
                if activation_function == 'relu' and l < self.total_layers - 1:
                    a = xp.where(z > 0, z, alpha * z)
                else:
                    a = activation_functions.sigmoid(z)
                activ.append(a)

            # Backward pass accumulating across batches
            # Output layer gradient
            delta = (activ[-1] - Yb) * activation_functions.sigmoid_prime(zs[-1])
            grads_b[-1] = xp.sum(delta, axis=(1, 2), keepdims=True)
            grads_w[-1] = xp.tensordot(delta, activ[-2], axes=([1, 2], [1, 2]))

            # Hidden layers
            for l in range(self.total_layers - 2, -1, -1):
                # Backpropagate delta
                delta = xp.tensordot(self.W[l + 1].T, delta, axes=([1], [0]))
                # Activation prime
                if activation_function == 'relu':
                    sp = xp.where(zs[l] > 0, 1, alpha)
                else:
                    sp = activation_functions.sigmoid_prime(zs[l])
                delta = delta * sp
                grads_b[l] = xp.sum(delta, axis=(1, 2), keepdims=True)
                grads_w[l] = xp.tensordot(delta, activ[l], axes=([1, 2], [1, 2]))

            # Parameter update (average over all samples)
            scale = learn_rate / (B * N)
            for i in range(self.total_layers):
                self.W[i] -= scale * grads_w[i]
                self.B[i] -= scale * grads_b[i]

            # Compute accuracy on original dataset
            final_activ = activ[-1].reshape(n_out, B * N)[:, :data_size]
            preds = xp.argmax(final_activ, axis=0)
            labels = xp.argmax(Y_mat, axis=0)
            acc = xp.mean(preds == labels)
            print(f"Epoch {epoch + 1} — Accuracy: {acc * 100:.2f}%")

    

class MNIST_loader:

    @staticmethod
    def load_file(dataFile, use_cupy=False):
        f = open(dataFile, 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
        f.close()
        if use_cupy:
            training_inputs = [cp.array(cp.reshape(x, (784, 1))) for x in training_data[0]]
            validation_data = [cp.array(x) for x in validation_data]
            test_inputs = [cp.array(x) for x in test_data]
        else:
            training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
            validation_data = [np.array(x) for x in validation_data]
            test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        #training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
        training_results = [MNIST_loader.vectorize_result(y, use_cupy=use_cupy) for y in training_data[1]]
        training_dataset = [training_inputs, training_results]
        test_dataset = [test_inputs, test_data[1]]  # Assuming test_data[1] contains labels
        return [training_dataset, test_dataset]
    
    @staticmethod
    def vectorize_result(j, use_cupy=False):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        if use_cupy:
            e = cp.zeros((10, 1), dtype=cp.float32)
        else:
            e = np.zeros((10, 1))
        e[j] = 1.0
        return e
        

def main():

    training_data, test_data = MNIST_loader.load_file(args.input_file, use_cupy=args.use_cupy)[:]
    training_x, training_y = training_data[:]
    test_x, test_y = test_data[:]

    activation_function = args.activation_funcion.lower()

    if activation_function != 'sigmoid' and activation_function != 'relu':
        print("Función de activación no válida. Funciones implementadas: sigmoid y relu.") 
        print("Usando 'sigmoid' por defecto.")
        activation_function = 'sigmoid'

    input_layer_size = len(training_x[0])
    output_layer_size = len(training_y[0])
    nn = neural_network(input_layer_size, output_layer_size, args.layers, activation_function, use_cupy=args.use_cupy)
    nn.train(training_x, training_y, activation_function, args.learn_rate, use_cupy=args.use_cupy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST data.')
    parser.add_argument("-i", "--input_file", dest='input_file', type=str, help='Path to the MNIST pickle file')
    parser.add_argument("-af", "--activation_function", dest='activation_funcion', default='sigmoid', type=str, help='Activation function. Options: sigmoid, relu')
    parser.add_argument("-r", "--learn_rate", dest='learn_rate', type=float, help='Learning rate for training')
    parser.add_argument('--cupy', dest="use_cupy", action='store_true', default=False, help='Use cupy for GPU acceleration')
    parser.add_argument('--layers', nargs='+', type=int, help='Array of integers for hidden layer sizes (e.g. --layers 64 32)')
    args = parser.parse_args()

    use_cupy = args.use_cupy
    if use_cupy:
        try:
            import cupy as cp
            if not cp.cuda.is_available():
                print("[AVISO] cupy solicitado pero no hay GPU disponible. Usando numpy.")
                use_cupy = False
        except ImportError:
            print("[AVISO] cupy no está instalado. Usando numpy.")
            use_cupy = False
    main()

