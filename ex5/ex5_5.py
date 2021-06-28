
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model


class ModelDecorator:
    def __init__(self, Model, alpha=None):
        self.__model = Model(alpha) if alpha else Model()

    def train(self, X, y):
        self.__model.fit(X, y)
        return self

    def predict(self, X):
        return self.__model.predict(X)

    def error(self, X, y):
        return np.mean((self.__model.predict(X) - y) ** 2)


def getModel(Model):
    return lambda x=None: ModelDecorator(Model, x)


def k_fold_validation(D_x, D_y, k_fold, Model, alpha_range):
    splits = np.split(np.arange(D_x.shape[0]), k_fold)
    train_error = np.zeros(len(alpha_range))
    validation_error = np.zeros(len(alpha_range))
    for k in range(k_fold):
        S = np.delete(D_x, splits[k], axis=0), np.delete(D_y, splits[k], axis=0)
        V = np.take(D_x, splits[k], axis=0), np.take(D_y, splits[k], axis=0)
        h = [Model(k).train(*S) for k in alpha_range]

        train_error += [h_i.error(*S) for h_i in h]
        validation_error += [h_i.error(*V) for h_i in h]

    return np.array(train_error) / k_fold, validation_error / k_fold


m = 50
folds = 5
Lasso = getModel(linear_model.Lasso)
Ridge = getModel(linear_model.Ridge)
Regression = getModel(linear_model.LinearRegression)


def main():
    X, y = datasets.load_diabetes(return_X_y=True)
    D = X[:m], y[:m]
    T = X[m:], y[m:]

    alpha_range = np.linspace(0.0001, 1.5, num=50)
    l_train_errors, l_validation_errors = k_fold_validation(D[0], D[1], folds, Lasso, alpha_range)
    r_train_errors, r_validation_errors = k_fold_validation(D[0], D[1], folds, Ridge, alpha_range)

    plt.plot(alpha_range, l_train_errors, label='train lasso error')
    plt.plot(alpha_range, l_validation_errors, label='validation lasso error')
    plt.plot(alpha_range, r_train_errors, label='train ridge error')
    plt.plot(alpha_range, r_validation_errors, label='validation ridge error')
    plt.xlabel('lambda')
    plt.ylabel('error')
    plt.title('Lasso & Ridge regularization CV')
    plt.legend()
    plt.show()

    best_ridge = alpha_range[np.argmin(r_validation_errors)]
    best_lasso = alpha_range[np.argmin(l_validation_errors)]
    print(f"ridge best lambda - {best_ridge}")
    print(f"lasso best alpha - {best_lasso}\n")

    print(f"ridge test error - {Ridge(best_ridge).train(*D).error(*T)}")
    print(f"lasso test error - {Lasso(best_lasso).train(*D).error(*T)}")
    print(f"linear regression test error - {Regression().train(*D).error(*T)}")


if __name__ == '__main__':
    main()
