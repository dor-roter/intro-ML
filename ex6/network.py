"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  
"""

#### Libraries
import numpy as np
import cupy as cp


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [cp.array(cp.random.randn(y, 1)) for y in sizes[1:]]
        self.weights = [cp.array(cp.random.randn(y, x))
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        a = cp.array(a)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(list(test_data))
        n = len(list(training_data))
        epochs_acc = []

        training_data = np.array(training_data, dtype=object)
        mini_batch_split = np.arange(mini_batch_size, n, mini_batch_size)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = np.split(training_data, mini_batch_split)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                hits = self.evaluate(test_data)
                epochs_acc.append(hits / n_test)
                print("Epoch {0}: {1} / {2}".format(
                    j, hits, n_test))
            else:
                print("Epoch {0} complete".format(j))
        return epochs_acc

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        xs, ys = mini_batch.T
        xs = cp.array(cp.vstack(xs).astype(np.float64).reshape((-1, self.sizes[0], 1)))
        ys = cp.array(cp.vstack(ys).astype(np.float64).reshape((-1, self.sizes[-1], 1)))
        delta_nabla_b, delta_nabla_w = self.backprop_batch(xs, ys)
        self.weights = [w - eta * cp.mean(nw, axis=0) for w, nw in zip(self.weights, delta_nabla_w)]
        self.biases = [b - eta * cp.mean(nb, axis=0) for b, nb in zip(self.biases, delta_nabla_b)]

    def backprop_batch(self, xs, ys):
        """
        xs each *column* is x_i
        :param xs:
        :param ys:
        :return:
        """
        nabla_b = [cp.zeros(b.shape) for b in self.biases]
        nabla_w = [cp.zeros(w.shape) for w in self.weights]

        # forward pass
        activation = xs
        activations = [xs]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            zs.append(w @ activation + b)
            activation = sigmoid(zs[-1])
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], ys) * f_sigmoid_prime(activations[-1])
        nabla_w[-1] = delta @ activations[-2].transpose((0, 2, 1))
        nabla_b[-1] = delta
        for t in range(self.num_layers - 2, 0, -1):
            delta = f_sigmoid_prime(activations[t]) * (self.weights[t].T @ delta)
            nabla_w[t - 1] = delta @ activations[t - 1].transpose((0, 2, 1))
            nabla_b[t - 1] = delta
        return nabla_b, nabla_w

    def evaluate(self, test_data, split=2):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        size = int(len(test_data) / split)
        mini_batch_split = np.arange(size, len(test_data), size)
        mini_batches = np.split(test_data, mini_batch_split)
        total = 0

        for mini_batch in mini_batches:
            xs, ys = np.array(mini_batch).T
            xs = cp.array(cp.vstack(xs).astype(np.float64).reshape((-1, self.sizes[0], 1)))
            ys = cp.array(ys.astype(np.int64))
            total += cp.sum(cp.argmax(self.feedforward(xs), axis=(1, 2)) == ys)
        return total

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return 2 * (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1 / (1 + cp.exp(-z))


def f_sigmoid_prime(sig):
    return sig * (1 - sig)


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return cp.exp(-z) / ((1 + cp.exp(-z)) ** 2)
