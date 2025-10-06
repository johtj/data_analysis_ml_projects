def plot_bias_variance_tradeoff_polynomial_degree_kfold(x_noise, y_noise, max_p=25, k = 5, use_intercept=True):
    """
    Plots and saves a figure of bias-variance tradeoff with different polynomial degrees (not using scikit-learn)

    Parameters
    ----------
    x_noise : numpy array shape (n)
    Data x coordinates

    y_noise : numpy array shape (n)
    Data y coordinates

    max_p: int
    The maximum polynomial degree to test.

    k: int
    k-fold number

    use_intercept: Bool
    Choose if using the intercept or not 
    """ 
    from sklearn.model_selection import KFold
    
    degrees = np.arange(1, max_p + 1, step=2)

    mses = np.zeros(degrees.shape)
    variances = np.zeros(degrees.shape)
    biases = np.zeros(degrees.shape)

    # make the k-fold object
    kf = KFold(n_splits=k, shuffle=True, random_state=2025)
    # loop through degrees
    for i, degree in enumerate(degrees):
        # make feature matrix
        X_noise = polynomial_features(x=x_noise, p=degree, intercept=use_intercept)

        preds = []
        # run the k-fold split and estimate OLS parameters
        for train_index, test_index in kf.split(X_noise):
            X_train, X_test = X_noise[train_index], X_noise[test_index]
            y_train, y_test = y_noise[train_index], y_noise[test_index]
            theta_OLS = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            y_tilde_test = X_test @ theta_OLS
            # save both the test data and predictions
            preds.append([y_test, y_tilde_test])

        # Shape: (n_points, test or pred, k-fold)
        preds = np.transpose(preds)
    
        mses[i] = MSE(preds[:, 0, :], preds[:, 1, :])
        variances[i] = variance(preds[:, 1, :])
        squared_bias_ = np.mean((preds[:, 0, :] - np.mean(preds[:, 1, :], axis=1)[:, None])**2)
        biases[i] = squared_bias_

    # plot and save figure 
    plt.figure(figsize=(6, 4))
    plt.plot(degrees, mses, label="MSE")
    plt.plot(degrees, variances, label="Variance")
    plt.plot(degrees, biases, label="Bias^2")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.xlabel("Polynomial degree", fontsize=12)
    plt.yscale("log")
    plt.savefig("Bias_variance_tradeoff_k_fold.png", bbox_inches='tight')
    plt.show()

np.random.seed(350)
# using noisy data
x_noise = np.linspace(-1, 1, num=100)
y_noise = (1 / (1 + 25 * x_noise**2)) + np.random.normal(0, 0.1, size=x_noise.size)
plot_bias_variance_tradeoff_polynomial_degree_kfold(x_noise, y_noise, max_p=24, k=5)