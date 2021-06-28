import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import time


MS = (50, 100, 300, 500)
ITERATIONS = 50

NEGATIVE = -1
POSITIVE = 1


def get_data(frac=0.25):
    """
    fetch MNIST data and prepare it
    :param frac: fraction of the dataset for testing
    :return: train test split of the prepared dataset
    """
    X, y = fetch_openml('mnist_784', return_X_y=True)
    y = y.astype(int)
    zero_one = (y == 0) | (y == 1)
    X, y = X[zero_one], y[zero_one]
    y = np.where(y == 0, NEGATIVE, POSITIVE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=frac)
    return (X_train, y_train), (X_test, y_test)


def sample_m(m, X, y):
    """
    sample m samples from X,y, make sure both tags are present in the sample
    :param m: number of samples
    :param X: all samples
    :param y: all tags
    :return: X,y for m samples
    """
    indexes = np.random.choice(X.shape[0], m)
    while np.all(y[indexes] == POSITIVE) or np.all(y[indexes] == NEGATIVE):
        indexes = np.random.choice(X.shape[0], m)
    return X[indexes], y[indexes]


def rearrange_data(X):
    """
    :param X: dataset of shape (-1, 28, 28)
    :return: reshaped matrix of shape (-1, 784)
    """
    return np.reshape(X, (-1, 784))


class TestedModel:
    """
    class to store and handle a single model testing
    """
    def __init__(self, name, model):
        """
        store model and name for testing
        :param name: model name for plotting
        :param model: the model
        """
        self.name = name
        self.model = model
        self.accuracy = {m: 0 for m in MS}
        self.running_time = {m: 0 for m in MS}
        self.__n = {m: 0 for m in MS}

    def add_accuracy_time(self, m, acc, running_time):
        """
        add accuracy and running time value to the means of those values
        :param m: the m for which we calculated the values
        :param acc: the accuracy value
        :param running_time: running time value
        """
        self.__n[m] += 1
        self.accuracy[m] += (1 / self.__n[m]) * (acc - self.accuracy[m])
        self.running_time[m] += (1 / self.__n[m]) * (running_time - self.running_time[m])


def tester():
    """
    run test and compare the models as described in q14
    """

    tested_models = [
        TestedModel('Logistic Regression', LogisticRegression()),
        TestedModel('Soft-SVM', SVC()),
        TestedModel('K-Nearest Neighbors', KNeighborsClassifier()),
        TestedModel('Decision Tree', DecisionTreeClassifier(max_depth=5)),
    ]
    train, (X_t, y_t) = get_data()
    for i, m in enumerate(MS):
        for j in range(ITERATIONS):
            X, y = sample_m(m, *train)

            for tested in tested_models:
                start = time.time()
                tested.model.fit(X, y)
                y_hat = tested.model.predict(X_t)
                acc = accuracy_score(y_t, y_hat)
                running_time = time.time() - start
                tested.add_accuracy_time(m, acc, running_time)

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    for tested in tested_models:
        ax1.plot(MS, [tested.accuracy[m] for m in MS], marker='.', label=tested.name)
        ax2.plot(MS, [tested.running_time[m] for m in MS], marker='.', label=tested.name)
    plt.legend()
    ax1.set_title('Training batch size vs. accuracy')
    ax2.set_title('Training batch size vs. running time')
    ax1.set(xlabel='m', ylabel='accuracy')
    ax2.set(xlabel='m', ylabel='running time')
    plt.show()


if __name__ == '__main__':
    tester()