import numpy as np
import matplotlib.pyplot as plt
f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


LOW = -3.2
HIGH = 2.2

MU = 0
SIGMA = 1

K = 15

M = 1500
SIZE = 500


def generate_data(low, high, sigma_2, m):
    X = np.random.uniform(low, high, m)
    y = f(X) + np.random.normal(MU, sigma_2, m)
    return X, y


def k_fold_validation(D_x, D_y, k_fold, Model):
    splits = np.split(np.arange(D_x.shape[0]), k_fold)
    train_error = np.zeros(K)
    validation_error = np.zeros(K)
    for k in range(k_fold):
        S = np.delete(D_x, splits[k]), np.delete(D_y, splits[k])
        V = np.take(D_x, splits[k]), np.take(D_y, splits[k])
        h = [Model(k + 1).train(*S) for k in range(K)]

        train_error += [h_i.error(*S) for h_i in h]
        validation_error += [h_i.error(*V) for h_i in h]

    return np.array(train_error) / k_fold, validation_error / k_fold


class PolyModel:
    def __init__(self, k):
        self.__k = k
        self.__w: np.array = None

    def predict(self, X):
        return np.vander(X, self.__k + 1, increasing=True) @ self.__w

    def error(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

    def train(self, X, y):
        X = np.vander(X, self.__k + 1, increasing=True)
        self.__w = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self


def items_b_c(D_x, D_y):
    S = D_x[:SIZE], D_y[:SIZE]
    V = D_x[SIZE:], D_y[SIZE:]
    h = [PolyModel(k + 1).train(*S) for k in range(K)]
    j = np.argmin([h_i.error(*V) for h_i in h])
    h_hat = h[j]
    return h_hat


def item_e(D_x, D_y):
    train_error, validation_error = k_fold_validation(D_x, D_y, 5, PolyModel)

    plt.plot(np.arange(K)+1, train_error, marker='.', label='train error')
    plt.plot(np.arange(K)+1, validation_error, marker='.', label='validation error')
    plt.xlabel('degree')
    plt.ylabel('error')
    plt.title('Train & Test errors vs polynomial fitting degree')
    plt.legend()
    plt.show()

    d_star = np.argmin(validation_error)
    print(f"best validation error is {validation_error[d_star]}")
    return d_star + 1


def main():
    X, y = generate_data(LOW, HIGH, SIGMA ** 2, M)
    D = X[:M - SIZE], y[:M - SIZE]
    T = X[M - SIZE:], y[M - SIZE:]

    h_hat = items_b_c(*D)
    d_star = item_e(*D)
    print(f"best polynomial degree fit is {d_star}")

    h_star = PolyModel(d_star).train(*D)
    test_error = h_star.error(*T)
    print(f"test error over T {test_error}")


if __name__ == '__main__':
    main()
