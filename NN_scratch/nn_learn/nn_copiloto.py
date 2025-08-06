import numpy as np
import random

class Network:
    def __init__(self, sizes):
        """
        sizes: lista con el número de neuronas en cada capa.
        Ejemplo: [784, 30, 10] para una red con 784 entradas, 1 capa oculta de 30 y 10 salidas.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Inicializa los sesgos y pesos con valores aleatorios
        self.B = [np.random.randn(y, 1) for y in sizes[1:]]
        self.W = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.Z = [np.zeros(i) for i in sizes[1:]]
        self.A = [np.zeros(i) for i in sizes[1:]]

    def sigmoid(self, z):
        """Función de activación sigmoide."""
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        """Calcula la salida de la red para una entrada a."""
        for b, w in zip(self.B, self.W):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def my_feedforward(self, x):
        
        
        self.Z[0] = np.dot(self.W[0], x) + self.B[0]
        #self.A[0] = np.maximum(0, self.Z[0])
        self.A[0] = self.sigmoid(self.Z[0])

        for i in range(2, self.num_layers-1):
            
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.B[i]
            #self.A[i] = np.maximum(0, self.Z[i]) #ReLU a todos los valores de Z
            self.A[i] = self.sigmoid(self.Z[i])
            
        
        self.Z[-1] = np.dot(self.W[-1], self.A[-2]) + self.B[-1]
        self.A[-1] = self.sigmoid(self.Z[-1])

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Entrena la red usando descenso de gradiente estocástico.
        training_data: lista de tuplas (x, y)
        epochs: número de épocas
        mini_batch_size: tamaño de los mini-batches
        eta: tasa de aprendizaje
        test_data: datos de prueba opcionales para evaluar el rendimiento
        """
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Época {j}: {self.evaluate(test_data)} / {len(test_data)} correctos")
            else:
                print(f"Época {j} completada")

    def update_mini_batch(self, mini_batch, eta):
        """
        Actualiza los pesos y sesgos usando un mini-batch y retropropagación.
        """
        nabla_b = [np.zeros(b.shape) for b in self.B]
        nabla_w = [np.zeros(w.shape) for w in self.W]
        for x, y in mini_batch:
            self.my_feedforward(x)
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.W = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.W, nabla_w)]
        self.B = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.B, nabla_b)]

    def backprop(self, x, y):
        """
        Retropropagación: calcula el gradiente de la función de coste respecto a pesos y sesgos.
        """
        nabla_b = [np.zeros(b.shape) for b in self.B]
        nabla_w = [np.zeros(w.shape) for w in self.W]
        # Propagación hacia adelante
        activation = x
        activations = [x] # lista para guardar todas las activaciones capa por capa
        zs = [] # lista para guardar todos los vectores z capa por capa
        for b, w in zip(self.B, self.W):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # Propagación hacia atrás
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.W[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def backpropagation(self, x, y, use_cupy=False):
        
        # x = self.X[0]
        # y = self.Y[0]
        
        # self.gradient_w = [np.zeros((self.layer_sizes[0], len(x)))]
        # self.gradient_w += [np.zeros((self.layer_sizes[i], self.layer_sizes[i-1])) for i in range(1, self.total_layers)]
        # self.gradient_b = [np.zeros(((self.layer_sizes[i]), 1)) for i in range(self.total_layers)]
        
        #Last layer
        nabla_b = [np.zeros(b.shape) for b in self.B]
        nabla_w = [np.zeros(w.shape) for w in self.W]
    
        aL = self.A[-1]
        dCo_daL = (aL-y)
        
        zL = self.Z[-1]
        daL_dzL = self.sigmoid_prime(zL)

        layer_error = daL_dzL * dCo_daL
        print(layer_error)
        exit()
        #layer_error = dCo_daL

        dzL_db = 1
        aL_left = self.A[-2]
        #dzL_dw = np.tile(aL_left, (len(self.A[-1]), 1))
        dCo_db = layer_error * dzL_db

        dCo_dw = np.dot(layer_error, aL_left.transpose())
        
        #print(dCo_dw.shape)
        #print(dCo_db.shape)
        #print()
        # print(self.gradient_b[0].shape)
        # print(dCo_db.shape)
        #exit()

        # Ensure correct array addition using numpy/cupy add
        
        nabla_b[-1] = dCo_db
        nabla_w[-1] = dCo_dw
        
        for i in range(2, self.num_layers):
            
            zL = self.Z[-i]
            # Cuando i = 0 recoge el valor de píxeles de la capa input. Else, recoge las a de la capa izquierda.
            aL_left = x if (i == self.num_layers-1) else self.A[i-1]

            #dzL_dw = np.tile(aL_left, (len(self.A[i]), len(aL_left)))
            
            dzL_db = 1
            daL_dzL = (zL > 0).astype(float)

            dzL_right_daL = self.W[-i+1]

            
            layer_error = np.dot(dzL_right_daL.transpose(), layer_error)
            layer_error = layer_error * daL_dzL

            dCo_db = layer_error * dzL_db
            dCo_dw = np.dot(layer_error, aL_left.transpose())

            #print(dCo_dw.shape)
            #print(dCo_db.shape)
            #print()

            
            nabla_b[-i] = dCo_db
            nabla_w[-i] = dCo_dw
        return (nabla_b, nabla_w)

    def sigmoid_prime(self, z):
        """Derivada de la función sigmoide."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def evaluate(self, test_data):
        """
        Evalúa la red en los datos de prueba y devuelve el número de aciertos.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Derivada de la función de coste respecto a la salida.
        """
        return (output_activations - y)

# Para comprobar que la clase funciona, puedes crear una red y pasarle datos aleatorios:
if __name__ == "__main__":
    net = Network([784, 30, 10])
    x = np.random.randn(784, 1)
    salida = net.feedforward(x)
    print("Salida de la red para una entrada aleatoria:", salida)