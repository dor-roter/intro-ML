import linear_model as lm

def feature_selection(X: pd.DataFrame, y: pd.DataFrame, thresh):
    """
    Evaluate for every feature it's pearson correlation
    against y and plot it against price for visualization
    :param thresh: min corr allowed
    :param X: dataframe of the design matrix
    :param y: response vector (price)
    :return: the two highest correlated features
    """
    S = ['intercept']
    while True:
        w, singular = lm.fit_linear_model(X[S].to_numpy(), y.to_numpy())
        y_hat = X[S] @ w
        z = y_hat - y
        corr = lm.pearson_corr(X, z)

        feature = corr.abs().idxmax()
        if abs(corr[feature]) < thresh:
            break

        y = z
        S.append(feature)

    S.append('price')
    return S