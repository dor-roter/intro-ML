import numpy as np
import matplotlib.pyplot as plt

import mnist_loader
import network


def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    hyper_param_selection(training_data, test_data)


def hyper_param_selection(training_data, test_data):
    batch_sizes = [5, 10, 20, 30, 50]
    accuracies = []
    epochs = np.arange(len(batch_sizes))

    for batch_size in batch_sizes:
        net = network.Network([784, 64, 64, 10])
        net.evaluate(test_data)
        accuracies.append(net.SGD(training_data, epochs=3, mini_batch_size=batch_size, eta=3, test_data=test_data))

    for i in range(1, len(batch_sizes)):
        plt.plot(epochs, accuracies[i], label=f"batch size {batch_sizes[i]}")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
