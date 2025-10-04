import numpy as np

def polynomial_features(x, p,intercept=False):

    """ 
    Generates a polynomial feature matrix with or without
    intercept, based on the values of x. 

    Returns
    -------
    X : numpy vector shape(n,p), if intercept shape(n,p+1)
        the resulting feature matrix of all polynomial combinations
        up to a given degree. Vandermonde format.
    

    Parameters
    ----------
    x : numpy vector shape(n)
        x values from dataset

    p : int
        number of degrees 

    intercept : Bool
        Bool to determine if intercept should be included or not:
        False : no intercept 
        True : include intercept
    """
    
    n = len(x)

    #handling the intercept column
    #to avoid branching in loop
    if intercept: 
        matrix_p = p+1
        start_col = 1
        i_offs = 0

        X = np.zeros((n, matrix_p))
        X[:,0] = np.ones(n)
   
    else:
        matrix_p = p
        start_col = 0
        i_offs = 1
    
        X = np.zeros((n, matrix_p))
    
    for i in range(start_col,matrix_p):
            X[:,i] = np.power(x,i+i_offs)
     
    return X


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





def scale_features_by_intercept_use(X_train, X_test, use_intercept):
    """
    Scales the feature matrix with or without intercept
    Keeps intercept column inscaled if use_intercept=True

    Returns
    -------

    X_train_scaled : numpy array shape (n,f)
        Standardized training feature matrix

    X_test_scaled : numpy array shape (n,f)
        Standardized test feature matrix

    Parameters
    ----------
    X_train : numpy array shape (n,f)
        Training feature matrix

    X_test : numpy array shape (n,f)
        Test feature matrix   
    
    use_intercept : Bool
        Bool to determine if intercept column should be included or not in scaling:
        False : no intercept 
        True : include intercept
   
    """
    if use_intercept == True:  
        X_train_scaled = X_train.copy()
        X_test_scaled  = X_test.copy()
        X_train_scaled_excluding_intercept, X_test_scaled_excluding_intercept, X_train_mean, X_train_std = standard_scaler(X_train[:, 1:], X_test[:, 1:])
        X_train_scaled[:, 1:] = X_train_scaled_excluding_intercept
        X_test_scaled[:, 1:]  = X_test_scaled_excluding_intercept
    else:
        X_train_scaled, X_test_scaled, X_train_mean, X_train_std = standard_scaler(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, X_train_mean, X_train_std
