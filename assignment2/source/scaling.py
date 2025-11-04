import autograd.numpy as np 

# from project 1
def standard_scaler(X_train, X_test):
    """
    Standardizes the feature matrix by removing the mean
    and scaling to unit variance.
    
    Verified to give identical results as sklearn.preprocessing.StandardScaler

    Returns
    -------
    X_train_scaled : numpy array shape (n,f)
        Standardized training feature matrix

    X_test_scaled : numpy array shape (n,f)
        Standardized test feature matrix
    
    X_mean : numpy array shape (n)
        Mean of columns in X

    X_std : numpy array shape (n)
        Standard deviation of columns in X

    Parameters
    ----------

    X_train : numpy array shape (n,f)
        Training feature matrix

    X_test : numpy array shape (n,f)
        Test feature matrix
    """
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)

    X_train_scaled = (X_train - X_train_mean) / X_train_std
    X_test_scaled = (X_test - X_train_mean) / X_train_std

    return X_train_scaled, X_test_scaled, X_train_mean, X_train_std



