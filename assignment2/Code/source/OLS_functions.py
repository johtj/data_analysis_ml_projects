# OLS functions from project 1

import autograd.numpy as np
import matplotlib.pyplot as plt


def OLS_parameters(X, y):
    """
        Calculates the optimal parameters, theta, using the 
        ordinary least squares method.  

        Theta_OLS = inv(X.T @ X) @ X.T @ y

        Returns
        -------
        theta : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method. 

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.
        
        y : numpy array shape (n)
            Y values of the data set.     
    """

    #calculate X^T*X and take the inverse
    XTX = X.T@X
    XTX_i = np.linalg.inv(XTX)

    #calculate X^T*y
    XT_y = X.T @ y
    
    #calculate theta
    theta = XTX_i @ XT_y
    
    return theta


def rescale_theta_intercept(coef_scaled, intercept_scaled, y_train_std, y_train_mean, X_train_std, X_train_mean, verbose):
    """
    Rescales theta and intercept back to original values

    Returns
    -------
    rescaled_intercept : float
        intercept at original scale

    rescaled_coef : numpy array shape (n)
        coefficients at original scale
        
    Parameters
    ----------
    coef_scaled : numpy array shape (n)
        scaled coefficients

    intercept_scaled : float
        scaled intercept
    
    y_train_std : float
        standard deviation of y_train

    y_train_mean : float
        mean of y_train

    X_train_std : float
        standard deviation of X_train

    X_train_mean : float
        mean of X_train

    verbose : Bool
        Include verbose output from function, default set to false
    """
    # Rescale coefficients
    rescaled_coef = [
        coef_scaled[i] * (y_train_std / X_train_std[i])
        for i in range(len(coef_scaled))
    ]
    rescaled_coef = np.array(rescaled_coef)
    # Rescale intercept
    rescaled_intercept = intercept_scaled * y_train_std + y_train_mean - np.sum(rescaled_coef[i] * X_train_mean[i] for i in range(len(rescaled_coef)))

    if verbose: print(f'Rescaled sklearn coef and interceot with own code\n Rescaled coef: {rescaled_coef}, rescaled intercept {rescaled_intercept}')

    return rescaled_coef, rescaled_intercept


def predict_y(X, theta):
    """
    Predicts y values from design matrix and theta

    Returns
    -------
    y_predict : numpy array shape (n)
        Predicted y values
        
    Parameters
    ----------
    X : numpy array shape (n,f)
        Feature matrix for the data, where n is the number
        of data points and f is the number of features.

    theta :  numpy array shape (n)
        coefficients for regression
    """
    return X @ theta

def rescale_y(predicted_y_scaled, y_train_std, y_train_mean):
    """
    Scales y values from scaled to original values

    Returns
    -------
    y_predict_rescaled : numpy array shape (n)
        Predicted y values at original values
        
    Parameters
    ----------
    predicted_y_scaled : numpy array shape (n)
        Predicted y values - scaled
    
    y_train_std : float
        standard deviation of y_train

    y_train_mean : float
        mean of y_train
    """
    return predicted_y_scaled * y_train_std + y_train_mean


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


def MSE(y, y_predict):
    """
    Calculates Mean Squared Error (MSE)
    between true and predicted values

    Returns
    -------
    mse : float
        MSE error value

    Parameters
    ----------

    y : numpy array shape (n)
        Y values of the data set. 
    
    y: numpy array shape (n)
        Predicted y values of the data set.
"""
    n = np.size(y_predict)
    mse = (1/n) * np.sum((y - y_predict)**2)
    return mse


def R2(y, y_predict):
    """
    Calculates R2 score
    between true and predicted values

    Returns
    -------
    r2 : float
        R2 score value

    Parameters
    ----------

    y : numpy array shape (n)
        Y values of the data set. 
    
    y: numpy array shape (n)
        Predicted y values of the data set.
"""
    ss_res = np.sum((y - y_predict)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def explore_polynomial_degree(X_train, X_test, y_train, y_test, p, use_intercept, verbose=False):
    """
    Explores the effect of polynomial degree on MSE and R2 for
    both training and test datasets using OLS regression.

    Returns
    -------

    polynomial_degree: list
        list of polynomial degrees explored
    
    mse_train: list
        list of MSE values for training data
    
    mse_test: list
        list of MSE values for test data

    r2_train: list
        list of R2 values for training data

    r2_test: list
        list of R2 values for test data

    Parameters
    ----------
    X_train : numpy array shape (n,f)
        Training feature matrix

    X_test : numpy array shape (n,f)
        Test feature matrix   

    y_train : numpy array shape (n)
        Training target vector

    y_test : numpy array shape (n)
        Test target vector
    
    p : int
        maximum polynomial degree to explore
    
    use_intercept : Bool
        Bool to determine if intercept should be included or not in regression:
        False : no intercept 
        True : include intercept
    
    verbose : Bool
        Include verbose output from function, default set to false
   
    """

    polynomial_degree = list()
    mse_train = list()
    mse_test = list()
    r2_train = list()
    r2_test = list()
    thetas = list() # thetas for polynomial degrees for plotting

    for degree in range(1, p+1):
        polynomial_degree.append(degree)

        # Extract the relevant columns from design matrix for the current degree
        X_train_sliced = X_train[:, :degree+1] 
        X_test_sliced = X_test[:, :degree+1]
        
        # OLS Regression
        theta_OLS = OLS_parameters(X_train_sliced, y_train)
        y_tilde_train = X_train_sliced @ theta_OLS
        y_tilde_test = X_test_sliced @ theta_OLS
        thetas.append(theta_OLS)

        # Calculate MSE for training and test data
        mse_train_OLS = MSE(y_train, y_tilde_train)
        mse_test_OLS = MSE(y_test, y_tilde_test)
        mse_train.append(mse_train_OLS)
        mse_test.append(mse_test_OLS)
        if verbose: print(f"Polynomial degree: {degree}, MSE_train_OLS: {mse_train_OLS}, MSE_test_OLS: {mse_test_OLS}")

        # Calculate R2 for training and test data
        r2_train_OLS = R2(y_train, y_tilde_train)
        r2_test_OLS = R2(y_test, y_tilde_test)
        r2_train.append(r2_train_OLS)
        r2_test.append(r2_test_OLS)
        if verbose: print(f"Polynomial degree: {degree}, R2_train_OLS: {r2_train_OLS}, R2_test_OLS: {r2_test_OLS}")

        if verbose: print('\n\n')

        # Sklearn Linear Regression without intercept for validation of code, test dataset only.
        # only for validation of own code        
        #from sklearn.linear_model import LinearRegression
        #model = LinearRegression(fit_intercept=use_intercept)
        #model.fit(X_train_sliced, y_train)
        #y_pred_sklearn = model.predict(X_test_sliced)
        #mse_sklearn = MSE(y_test, y_pred_sklearn)
        #r2_sklearn = R2(y_test, y_pred_sklearn)

        #if verbose:
            #print(f"Polynomial degree: {degree}, Sklearn test R2: {r2_sklearn}, Sklearn test MSE: {mse_sklearn}")
            #print(f"Polynomial degree: {degree}, R2 test: Own - sklearn {r2_test_OLS - r2_sklearn}, MSE test: Own - sklearn {mse_test_OLS - mse_sklearn}")
            #print(f"Polynomial degree: {degree}, Coef: {model.coef_}, intercept: {model.intercept_}")
            #print('\n') # just to add line shift between different degrees in output
    
    return polynomial_degree, mse_train, mse_test, r2_train, r2_test, thetas



def plot_mse(method,n,x_axis_data, mse_train, mse_test,x_label,title='remember title',save=False,fname=""):
    """
    Plots the Mean Squared Error (MSE) for different polynomial degrees.
    
    Returns
    -------
    Saves and shows a plot of MSE for training and test sets.

    Parameters
    ----------
    regression_method : string
        Type of regression for plotting

    degree : int
        number of polynomials in regression
    
    n_datapoints : int
        number of data points

    polynomial degree : int
        polynomial degree for regression

    mse_train : list
        list of MSE values for training set
    
    mse_test : list
        list of MSE values for test set

    noise : Bool
        Bool to determine if noise is included or not in dataset:
    """

   
    # removed title from plot, but keep code in case needed later
    #plt.title(text)  
    plt.plot(x_axis_data, mse_train, 'o-',label=f'MSE train {method}')
    plt.plot(x_axis_data, mse_test, 'o-', label=f'MSE test {method}')

    plt.xlabel(x_label)
    plt.ylabel('Mean Squared Error',fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend()
    plt.title(title)
    
    if save: plt.savefig("../figures/"+fname, bbox_inches='tight')

    plt.show()
    plt.close()

