import numpy as np
import matplotlib.pyplot as plt

import models


D = 2
BIAS = 0.1
W = np.array([.3, -.5])
F = lambda X: np.sign((X @ W) + BIAS)
MS = (5, 10, 15, 25, 70)


def hyperplane_line(x, w, bias):
    """
    calculate the hyperplane f(x) value based on x, w and the bias where
    the hyperplane is described by (x*w0 + y*w1 + b)
    :param x: the point for which we want to calculate f(x)
    :param w: the hyperplane vertical
    :param bias: the hyperplane intercept
    :return: the y value based on x
    """
    return (x * w[0] + bias) / (-w[1])


def draw_points(m):
    """
    generate m data point based on the described distribution
    :param m: number of samples
    :return: X,y dataset
    """
    X = np.random.multivariate_normal(np.zeros(D), np.eye(D), m)
    return X, F(X)


def sample_d(m):
    """
    get m samples from D (draw_points) with both labels present
    :param m: number of samples
    :return: X, y dataset
    """
    if m <= 2:
        return
    X, y = draw_points(m)
    while len(np.unique(y)) < 2:
        X, y = draw_points(m)
    return X, y


def plot():
    """
    plot the hyperplanes portrayed by each model for m samples sampled from the distribution m
    """
    ROWS = 2
    COLS = 3

    fig, axs = plt.subplots(ROWS, COLS, figsize=(15, 10))
    scatter = None
    for i, m in enumerate(MS):
        X, y = draw_points(m)
        x_lim = np.array([min(X[:, 0]), max(X[:, 0])])
        ax = axs[i // COLS, i % COLS]

        perceptron = models.Perceptron()
        perceptron.fit(X, y)
        svm = models.SVM()
        svm.fit(X, y)

        scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.plot(x_lim, hyperplane_line(x_lim, W, BIAS), label='f')
        ax.plot(x_lim, hyperplane_line(x_lim, perceptron.get_w(), perceptron.get_b()), label='Perceptron')
        ax.plot(x_lim, hyperplane_line(x_lim, svm.get_w(), svm.get_b()), label='SVM')

        ax.set(xlabel='x', ylabel='y')
        ax.set_title(f"Data for m={m}")
        ax.legend()

    plt.legend(handles=scatter.legend_elements()[0], labels=('Negative', 'Positive'), loc=4)
    axs[-1, -1].axis('off')
    fig.tight_layout()
    plt.show()


class TestedModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.accuracy = {m: 0 for m in MS}
        self.__n = {m: 0 for m in MS}

    def reset_accuracy(self, m):
        self.accuracy[m] = 0
        self.__n[m] = 0

    def add_accuracy(self, m, acc):
        self.__n[m] += 1
        self.accuracy[m] += (1 / self.__n[m]) * (acc - self.accuracy[m])


def test():
    """
    test Perceptron, SVM and LDA accuracy
    """
    tested_models = [
        TestedModel('Perceptron', models.Perceptron()),
        TestedModel('SVM', models.SVM()),
        TestedModel('LDA', models.LDA()),
    ]

    k = 10000
    iterations = 500
    for i, m in enumerate(MS):
        for j in range(iterations):
            X, y = sample_d(m)
            X_t, y_t = sample_d(k)

            for tested in tested_models:
                tested.model.fit(X, y)
                score = tested.model.score(X_t, y_t)
                tested.add_accuracy(m, score['accuracy'])

    plt.figure()
    for tested in tested_models:
        plt.plot(MS, [tested.accuracy[m] for m in MS], marker='.', label=tested.name)
    plt.legend()
    plt.title('Training batch size vs. accuracy')
    plt.xlabel('m')
    plt.ylabel('accuracy')
    plt.show()

    for tested in tested_models:
        print(tested.name, tested.accuracy)


if __name__ == '__main__':
    plot()
    test()
