from abc import ABC, abstractmethod
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

POSITIVE = 1
NEGATIVE = -1


class ModelException(Exception):
    """ exception type for models """
    pass


class Model(ABC):
    """
    single model abstract class
    """
    @abstractmethod
    def fit(self, X, y):
        """
        fit the model using provided samples
        :param X: samples
        :param y: labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        predict value for the provided samples
        :param X: samples to predict for
        :return: y_hat predicted labels
        """
        pass

    def score(self, X, y):
        """
        score a model by testing it against the provided dataset
        :param X: samples
        :param y: actual label
        :return: score dictionary
        """
        num_samples = X.shape[0]
        if num_samples == 0:
            raise ModelException()

        y_hat = self.predict(X)
        y = y.reshape((-1))
        FP = np.count_nonzero((y_hat == POSITIVE) & (y == NEGATIVE))
        FN = np.count_nonzero((y_hat == NEGATIVE) & (y == POSITIVE))
        TN = np.count_nonzero((y_hat == y) & (y == NEGATIVE))
        TP = np.count_nonzero((y_hat == y) & (y == POSITIVE))
        N = max(FP + TN, 1)
        P = max(FN + TP, 1)

        return {
            'num_samples': num_samples,
            'error': (FP + FN) / num_samples,
            'accuracy': (TP + TN) / num_samples,
            'FPR': FP / N,
            'FNR': FN / P,
            'precision': TP / (TP + FP),
            'specificity': TN / N
        }


class Perceptron(Model):
    """
    Perceptron model
    """
    def __init__(self):
        """
        constructor
        """
        self.__w = None
        self.__ready = False

    @staticmethod
    def __add_b(X):
        """
        private static method to add intercept column to the dataset
        :param X: samples
        :return: samples with added intercept
        """
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        """
        fit the model using provided samples
        :param X: samples
        :param y: labels
        """
        X = self.__add_b(X)

        self.__w = np.zeros(X.shape[1])
        check_misses = ((X @ self.__w) * y) <= 0
        i = np.argmax(check_misses)
        while check_misses[i]:
            self.__w += y[i] * X[i]
            check_misses = ((X @ self.__w) * y) <= 0
            i = np.argmax(check_misses)

        self.__ready = True

    def predict(self, X):
        """
        predict value for the provided samples
        :param X: samples to predict for
        :return: y_hat predicted labels
        """
        if not self.__ready:
            raise ModelException()

        X = self.__add_b(X)
        return np.where((X @ self.__w.T) > 0, POSITIVE, NEGATIVE)

    def get_w(self):
        """
        :return: get the weights vector for the model
        """
        return self.__w[1:]

    def get_b(self):
        """
        :return: get the intercept of the model
        """
        return self.__w[0]


class LDA(Model):
    """
    LDA model
    """
    def __init__(self):
        """
        constructor
        """
        self.__ready = False
        self.__mu = None
        self.__sigma = None
        self.__sigma_inv = None
        self.__p = None
        self.__const = None

    @staticmethod
    def __to_classes(mat):
        """
        private method to convert 0,1 to -1,1 labels
        :param mat:
        :return:
        """
        return np.where(mat == 0, NEGATIVE, POSITIVE)

    def predict(self, X):
        """
        predict value for the provided samples
        :param X: samples to predict for
        :return: y_hat predicted labels
        """
        if not self.__ready:
            raise ModelException()
        return self.__to_classes(
            np.argmax([X @ self.__sigma_inv @ self.__mu[i] + self.__const[i] for i in range(2)], axis=0))

    def fit(self, X, y):
        """
        fit the model using provided samples
        :param X: samples
        :param y: labels
        """
        m = len(y)
        NEG = y == NEGATIVE
        POS = y == POSITIVE
        P = X[POS]
        N = X[NEG]

        self.__ready = True
        self.__mu = np.array((np.mean(N, axis=0), np.mean(P, axis=0)))
        self.__p = np.array((np.mean(NEG), np.mean(POS)))
        sigma = (1 / (m - 1)) * ((N - self.__mu[0]).T @ (N - self.__mu[0]) + (P - self.__mu[1]).T @ (P - self.__mu[1]))
        self.__sigma_inv = np.linalg.inv(sigma)
        self.__const = [(-.5*self.__mu[i].T @ self.__sigma_inv @ self.__mu[i] + np.log(self.__p[i])) for i in range(2)]


class SVM(SVC, Model):
    """
    Hard-SVM model
    """
    def __init__(self):
        """
        constructor
        """
        SVC.__init__(self, C=1e10, kernel='linear')

    def score(self, X, y, sample_weight=None):
        """
        score a model by testing it against the provided dataset
        :param X: samples
        :param y: actual label
        :param sample_weight: not needed
        :return: score dictionary
        """
        return Model.score(self, X, y)

    def get_w(self):
        """
        :return: get the weights vector for the model
        """
        return self.coef_[0, :]

    def get_b(self):
        """
        :return: get the intercept of the model
        """
        return self.intercept_


class Logistic(LogisticRegression, Model):
    """
    logistic regression model
    """
    def __init__(self):
        """
        constructor
        """
        LogisticRegression.__init__(self, solver='liblinear')

    def score(self, X, y, sample_weight=None):
        """
        score a model by testing it against the provided dataset
        :param X: samples
        :param y: actual label
        :param sample_weight: not needed
        :return: score dictionary
        """
        return Model.score(self, X, y)


class DecisionTree(DecisionTreeClassifier, Model):
    """
    Decision Tree model
    """
    def __init__(self):
        """
        constructor
        """
        DecisionTreeClassifier.__init__(self, max_depth=5)

    def score(self, X, y, sample_weight=None):
        """
        score a model by testing it against the provided dataset
        :param X: samples
        :param y: actual label
        :param sample_weight: not needed
        :return: score dictionary
        """
        return Model.score(self, X, y)
