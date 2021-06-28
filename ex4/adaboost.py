"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np
import matplotlib.pyplot as plt

import ex4_tools

MAX_T = 500


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = np.empty(m)
        D.fill(1/m)

        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            y_hat = self.h[t].predict(X)
            eps = np.sum(D * (y_hat != y))
            self.w[t] = 0.5 * np.log(1./eps - 1)
            D *= np.exp(-y*self.w[t]*y_hat)
            D /= np.sum(D)
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predictions = [h.predict(X) for h in self.h[:max_t]]
        return np.sign(self.w[:max_t] @ predictions)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        return np.count_nonzero(self.predict(X, max_t) != y) / X.shape[0]


def q13(train, test):
    adaboost = AdaBoost(ex4_tools.DecisionStump, MAX_T)
    adaboost.train(*train)

    plt.figure()
    ts = np.arange(1, MAX_T + 1)
    plt.plot(ts, [adaboost.error(*train, t) for t in ts], label='train error')
    plt.plot(ts, [adaboost.error(*test, t) for t in ts], label='test error')
    plt.xlabel("T (number of models)")
    plt.ylabel("error")
    plt.title("AdaBoost - error vs ensemble size")
    plt.legend()
    plt.show()


def q14(train, test):
    T = (5, 10, 50, 100, 200, 500)
    adaboost = AdaBoost(ex4_tools.DecisionStump, MAX_T)

    X, y = train
    X_t, y_t = test
    adaboost.train(X, y)

    plt.figure()
    for i, t in enumerate(T):
        plt.subplot(2, 3, i + 1)
        ex4_tools.decision_boundaries(adaboost, X_t, y_t, t)
    plt.title("Decisions Boundaries")
    plt.show()


def q15(train, test):
    T = (5, 10, 50, 100, 200, 500)
    adaboost = AdaBoost(ex4_tools.DecisionStump, MAX_T)
    adaboost.train(*train)

    errors = [adaboost.error(*test, t) for t in T]
    best = np.argmin(errors)
    plt.figure()
    ex4_tools.decision_boundaries(adaboost, *test, T[best])
    plt.title(f"Best classifier - T={T[best]}, error={errors[best]}")
    plt.show()


def q16(train):
    adaboost = AdaBoost(ex4_tools.DecisionStump, MAX_T)
    D = adaboost.train(*train)
    D = D / np.max(D) * 10
    plt.figure()
    ex4_tools.decision_boundaries(adaboost, *train, MAX_T, weights=D)
    plt.title("Weighted Training Set")
    plt.show()


def main():
    noises = (0, 0.01, 0.4)
    for noise in noises:
        train = ex4_tools.generate_data(5000, noise)
        test = ex4_tools.generate_data(200, noise)

        q13(train, test)
        q14(train, test)
        q15(train, test)
        q16(train)


if __name__ == '__main__':
    main()
