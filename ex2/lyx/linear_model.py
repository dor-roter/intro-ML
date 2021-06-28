import numpy as np
import pandas as pd
from scipy.linalg import svdvals

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=2)

TEST_FRAC = 0.25
DATETIME_FORMAT = "%Y%m%dT000000"

GT_ZERO_VALUES = ['price', 'zipcode']
GE_ZERO_VALUES = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above',
                  'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

CATEGORICAL = {'intercept', 'zipcode', 'yr_built', 'yr_renovated', 'location'}
CATEGORICAL = '|'.join(CATEGORICAL)


def round_to_decade(x):
    """
    round up the decade
    """
    return int(round(x / 10.0)) * 10


def fit_linear_model(X, y):
    """
    :param X: a m-by-d design matrix
    :param y: a R^m response vector
    :returns: asf
    """
    w = np.linalg.pinv(X) @ y
    s = svdvals(X)
    return w, s


def predict(X, w):
    """
    Predict a design matrix using a provided model
    :param X: a m-by-d design matrix
    :param w: a R^d weights vector
    :return: y hat vector of m predictions
    """
    return X @ w


def mse(y, y_hat):
    """
    calculate the MSE of the Y_hat prediction
    :param y: R_m response vector
    :param y_hat: R_m prediction vector
    :return: the MSE of Y_hat
    """
    return np.mean((y - y_hat) ** 2)


def load_data(path):
    """
    load the housing data from the provided path and preform preprocessing
    :param path: the housing dataset in csv format path
    :return: the preprocessed dataframe
    """
    df = pd.read_csv(path)

    # drop corrupted data
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # validate ranges
    df = df[df['waterfront'].isin(range(2))]
    df = df[df['view'].isin(range(5))]
    df = df[df['condition'].isin(range(1, 6))]
    df = df[df['grade'].isin(range(1, 12))]

    # validate continues values ranges
    for col in GT_ZERO_VALUES:
        df = df[df[col] > 0]
    for col in GE_ZERO_VALUES:
        df = df[df[col] >= 0]

    # cluster years to decades and add is_new flag for really new places
    sale_year = pd.to_datetime(df['date'], format=DATETIME_FORMAT, errors='coerce').dt.year
    df['is_new'] = ((sale_year-df[['yr_built', 'yr_renovated']].max(axis=1)) <= 5).astype(int)
    df['yr_built'] = df['yr_built'].apply(round_to_decade)
    df['yr_renovated'] = df['yr_renovated'].apply(round_to_decade)

    # cluster lat/long to ~10km approximation
    df['lat'] = df['lat'].round(decimals=1)
    df['long'] = df['long'].round(decimals=1)
    df['location'] = list(zip(df['lat'], df['long']))
    df.drop(['lat', 'long'], axis=1, inplace=True)

    # is_basement flag
    df['is_basement'] = (df['sqft_basement'] >= 1).astype(int)

    # # one hot encode zip, location and decades
    df = pd.get_dummies(df, columns=['zipcode', 'location', 'yr_built', 'yr_renovated'])

    # remove outliers/anomalies
    df = df[df['bedrooms'] < 30]

    # add 1,...,1 intercept column
    df.insert(0, "intercept", np.ones(df.shape[0]), True)

    # drop not unuseful / uncorrelated
    df.drop(['id', 'date'], axis=1, inplace=True)

    return df


def plot_singular_values(singular):
    """
    plot the singular values provided as a scree-plot
    :param singular: array of singular values
    """
    values_number = np.arange(len(singular)) + 1
    plt.plot(values_number, singular, 'ro-', linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Component Number")
    plt.ylabel("Singular Value")
    plt.yscale('log')
    plt.grid()
    plt.show()


def get_training_test_split(df, frac):
    """
    Split the dataset into a training set and a test set
    by the provided fraction
    :param df: the dataframe
    :param frac: the required testing set proportion (percentage)
    :return: two tuple of X,y as numpy arrays (for each of the data sets)
    """
    test = df.sample(frac=frac)
    train = df.drop(test.index)

    X = train.drop(['price'], axis=1)
    y = train['price']
    X_test = test.drop(['price'], axis=1)
    y_test = test['price']

    train = X.to_numpy(), y.to_numpy()
    test = X_test.to_numpy(), y_test.to_numpy()
    return train, test


def r2_score(y, y_hat):
    """
    get r2 score for the model
    :param y: response vector
    :param y_hat: prediction vector by the model
    :return: r2 score of the prediction made by the model
    """
    y_bar = np.mean(y)  # or sum(y)/len(y)
    ssreg = np.sum((y_hat - y_bar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - y_bar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    return {'determination': (ssreg / sstot)}


def fit_and_test(X, y, X_test, y_test):
    """
    fit the model gradually on increasing sizes of the training set
    and plot the MSE against the test set
    :param X: training design matrix
    :param y: training response matrix
    :param X_test: test design matrix
    :param y_test: test response matrix
    """
    ps = np.arange(1, 101)
    mses, w = [], []
    for p in ps:
        data_count = int(X.shape[0] * (p / 100))
        w, singular = fit_linear_model(X[:data_count, :], y[:data_count])
        mses.append(mse(y_test, predict(X_test, w)))

    print(r2_score(y_test, predict(X_test, w)))
    plt.plot(ps, mses)
    plt.title("MSE vs training set size")
    plt.ylabel("MSE")
    plt.xlabel("p% of the training set used (%)")
    plt.show()


def pearson_corr(X, y) -> pd.Series:
    """
    compute pearson correlation of each feature and the response vector
    :param X: design  matrix
    :param y: response vector
    :return: pearson correlation of X's features and y
    """
    unbiased_cov = np.cov(X, y, ddof=True, rowvar=False)[-1, :-1]
    unbiased_stds = np.std(y, ddof=True) * np.std(X, axis=0, ddof=True)
    return unbiased_cov / unbiased_stds


def feature_evaluation(X, y):
    """
    Evaluate for every feature it's pearson correlation
    against y and plot it against price for visualization
    :param X: dataframe of the design matrix
    :param y: response vector (price)
    :return: the two highest correlated features
    """
    corr = pearson_corr(X, y)
    for col in X:
        plt.figure()

        plt.scatter(X[col], y, marker='.', s=20, linewidth=0.2, c='orange', edgecolor='k', label=col)
        m = np.polyfit(X[col], y, 1)
        p = np.poly1d(m)
        plt.plot(X[col], p(X[col]), "r--", alpha=0.5, label='trend line')

        plt.title(f"Feature '{col}' against response \n Pearson correlation {round(corr[col], 2)}")
        plt.xlabel(col)
        plt.ylabel('price')
        plt.legend()
        plt.show()

    corr.sort_values(inplace=True)
    best, worst = corr[-2:], corr[0:1]
    print("highest correlated features: ")
    print("--------------------------")
    print("best:\n-----")
    print(best)
    print("worst:\n-----")
    print(worst)


def main():
    # load the data
    df = load_data('kc_house_data.csv')

    # analyze the design matrix (q14, q17)
    df_none_cat = df.loc[:, ~df.columns.str.contains(CATEGORICAL)]
    X = df_none_cat.drop('price', 1)
    plot_singular_values(svdvals(X))
    feature_evaluation(X, df_none_cat['price'])

    # fit and test (q16)
    train, test = get_training_test_split(df, TEST_FRAC)
    fit_and_test(*train, *test)


if __name__ == '__main__':
    main()
