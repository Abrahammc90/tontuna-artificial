import sys
import argparse
import pickle

# GPU acceleration support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class activation_functions:
    @staticmethod
    def sigmoid(x, xp):
        return 1 / (1 + xp.exp(-x))

    @staticmethod
    def sigmoid_prime(x, xp):
        s = activation_functions.sigmoid(x, xp)
        return s * (1 - s)

    @staticmethod
    def relu(x, xp):
        return xp.maximum(0, x)

    @staticmethod
    def relu_prime(x, xp):
        return (x > 0).astype(xp.float32)

    @staticmethod
    def leaky_relu(x, xp, alpha=0.01):
        return xp.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_prime(x, xp, alpha=0.01):
        return xp.where(x > 0, 1, alpha).astype(xp.float32)

class neural_network:
    def __init__(self, input_size, output_size, hidden_sizes, activation, use_cupy=False):
        # choose backend
        if use_cupy and GPU_AVAILABLE:
            self.xp = cp
        else:
            self.xp = __import__('numpy')
            use_cupy = False
        self.use_cupy = use_cupy
        self.activation = activation

        # layer dimensions
        self.sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.sizes) - 1
        # weights and biases
        self.W = []
        self.B = []
        for i in range(self.num_layers):
            fan_in = self.sizes[i]
            fan_out = self.sizes[i+1]
            if activation == 'relu' or activation == 'leaky_relu':
                scale = (2 / fan_in) ** 0.5
                W = self.xp.random.randn(fan_out, fan_in, dtype=self.xp.float32) * scale
                b = self.xp.zeros((fan_out,1), dtype=self.xp.float32) + 0.01
            else:
                W = self.xp.random.randn(fan_out, fan_in, dtype=self.xp.float32)
                b = self.xp.zeros((fan_out,1), dtype=self.xp.float32)
            self.W.append(W)
            self.B.append(b)

    def _activate(self, z):
        xp = self.xp
        if self.activation == 'sigmoid':
            return activation_functions.sigmoid(z, xp)
        elif self.activation == 'relu':
            return activation_functions.relu(z, xp)
        elif self.activation == 'leaky_relu':
            return activation_functions.leaky_relu(z, xp)
        else:
            raise ValueError(f"Unsupported activation {self.activation}")

    def _activate_prime(self, z):
        xp = self.xp
        if self.activation == 'sigmoid':
            return activation_functions.sigmoid_prime(z, xp)
        elif self.activation == 'relu':
            return activation_functions.relu_prime(z, xp)
        elif self.activation == 'leaky_relu':
            return activation_functions.leaky_relu_prime(z, xp)
        else:
            raise ValueError(f"Unsupported activation {self.activation}")

    def feedforward(self, x):
        a = x
        for W, b in zip(self.W[:-1], self.B[:-1]):
            z = W.dot(a) + b
            a = self._activate(z)
        # output layer uses sigmoid
        z = self.W[-1].dot(a) + self.B[-1]
        a = activation_functions.sigmoid(z, self.xp)
        return a

    def backprop(self, x, y):
        xp = self.xp
        # forward
        activations = [x]
        zs = []
        a = x
        for W, b in zip(self.W[:-1], self.B[:-1]):
            z = W.dot(a) + b
            zs.append(z)
            a = self._activate(z)
            activations.append(a)
        # output
        z = self.W[-1].dot(a) + self.B[-1]
        zs.append(z)
        a = activation_functions.sigmoid(z, xp)
        activations.append(a)
        # grads init
        nabla_w = [xp.zeros_like(W) for W in self.W]
        nabla_b = [xp.zeros_like(b) for b in self.B]
        # output error
        delta = (a - y) * activation_functions.sigmoid_prime(z, xp)
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].T)
        # backpropagate
        for l in range(2, self.num_layers+1):
            z = zs[-l]
            sp = self._activate_prime(z)
            delta = self.W[-l+1].T.dot(delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta.dot(activations[-l-1].T)
        return nabla_w, nabla_b

    def update_batch(self, batch_x, batch_y, lr):
        xp = self.xp
        nabla_w_sum = [xp.zeros_like(W) for W in self.W]
        nabla_b_sum = [xp.zeros_like(b) for b in self.B]
        for x, y in zip(batch_x, batch_y):
            nw, nb = self.backprop(x, y)
            nabla_w_sum = [nw_sum + dw for nw_sum, dw in zip(nabla_w_sum, nw)]
            nabla_b_sum = [nb_sum + db for nb_sum, db in zip(nabla_b_sum, nb)]
        m = len(batch_x)
        self.W = [W - (lr/m)*nw for W, nw in zip(self.W, nabla_w_sum)]
        self.B = [b - (lr/m)*nb for b, nb in zip(self.B, nabla_b_sum)]

    def train(self, X, Y, lr, epochs, batch_size):
        xp = self.xp
        n = len(X)
        for e in range(epochs):
            perm = xp.random.permutation(n)
            X_sh = [X[i] for i in perm]
            Y_sh = [Y[i] for i in perm]
            for k in range(0, n, batch_size):
                xb = X_sh[k:k+batch_size]
                yb = Y_sh[k:k+batch_size]
                self.update_batch(xb, yb, lr)
            correct = sum(int(xp.argmax(self.feedforward(x)) == xp.argmax(y)) for x, y in zip(X, Y))
            print(f"Epoch {e+1}/{epochs} - Accuracy: {correct}/{n} = {correct/n*100:.2f}%")

class MNIST_loader:
    @staticmethod
    def load(data_file, use_cupy=False):
        with open(data_file, 'rb') as f:
            train, _, test = pickle.load(f, encoding='bytes')
        X_train, y_train = train
        X_test, y_test = test
        if use_cupy and GPU_AVAILABLE:
            xp = cp
            X_train = [xp.array(x.reshape(784,1), dtype=xp.float32) for x in X_train]
            y_train = [MNIST_loader.vectorize(y, xp) for y in y_train]
            X_test = [xp.array(x.reshape(784,1), dtype=xp.float32) for x in X_test]
        else:
            import numpy as np
            xp = np
            X_train = [xp.array(x.reshape(784,1)) for x in X_train]
            y_train = [MNIST_loader.vectorize(y, xp) for y in y_train]
            X_test = [xp.array(x.reshape(784,1)) for x in X_test]
        return X_train, y_train, X_test, y_test

    @staticmethod
    def vectorize(j, xp):
        e = xp.zeros((10,1), dtype=xp.float32)
        e[j] = 1.0
        return e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST data.')
    parser.add_argument("-i", "--input_file", dest='input_file', type=str, help='Path to the MNIST pickle file')
    parser.add_argument("-af", "--activation_function", dest='activation_function', default='sigmoid', type=str, help='Activation function. Options: sigmoid, relu')
    parser.add_argument("-r", "--learn_rate", dest='learn_rate', type=float, help='Learning rate for training')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=10, help='Size of the batches')
    parser.add_argument('-cp', '--cupy', dest="use_cupy", action='store_true', default=False, help='Use cupy for GPU acceleration')
    parser.add_argument('-hl', '--hidden_layers', dest="hidden_layers", nargs='+', type=int, help='Array of integers for hidden layer sizes (e.g. --layers 64 32)')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = MNIST_loader.load(args.input_file, args.use_cupy)
    input_size = 784
    output_size = 10
    hidden = args.hidden_layers  # customize or parse more args
    nn = neural_network(input_size, output_size, hidden, args.activation_function, use_cupy=args.use_cupy)
    nn.train(X_train, y_train, args.learn_rate, args.epochs, args.batch_size)

    # final test accuracy
    correct = sum(int(nn.xp.argmax(nn.feedforward(x)) == nn.xp.argmax(y)) for x,y in zip(X_test,y_test))
    total = len(X_test)
    print(f"Test accuracy: {correct}/{total} = {correct/total*100:.2f}%")
