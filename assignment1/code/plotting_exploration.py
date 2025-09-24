import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from main_methods import OLS_parameters,Ridge_parameters
from errors import MSE,R2

def plot_mse(n_datapoints, x_axis, x_axis_label, mse_train, mse_test, noise=False):
    """
    Plots the Mean Squared Error (MSE) for different polynomial degrees.
    
    Returns
    -------
    Saves and shows a plot of MSE for training and test sets.

    Parameters
    ----------

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

    if noise:
        text = f'MSE for Different {x_axis_label} with Noise\nNumber of data points: {n_datapoints}'
        filename = f'MSE for Different {x_axis_label} with Noise - Number of data points {n_datapoints}.png'
        plt.title(text)
    else:
        text = f'MSE for Different {x_axis_label} without Noise\nNumber of data points: {n_datapoints}'
        filename = f'MSE for Different {x_axis_label} without Noise - Number of data points {n_datapoints}.png'
        plt.title(text)
    plt.plot(x_axis, mse_train, label='MSE train')
    plt.plot(x_axis, mse_test, label='MSE test')
    plt.xlabel(x_axis_label)
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()



def plot_r2(n_datapoints, x_axis,x_axis_label, r2_train, r2_test, noise=False):
    """
    Plots the R2 Score for different polynomial degrees.
    
    Returns
    -------
    Saves and shows a plot of R2 Score for training and test sets.

    Parameters
    ----------

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
        
    if noise:
        plt.title(f'R2 Score for Different {x_axis_label} with Noise\nNumber of data points: {n_datapoints}')
        filename = f'R2 for Different {x_axis_label} with Noise - Number of data points {n_datapoints}.png'
    else:
        plt.title(f'R2 Score for Different {x_axis_label} without Noise\nNumber of data points: {n_datapoints}')
        filename = f'R2 for Different {x_axis_label} without Noise - Number of data points {n_datapoints}.png'
    plt.plot(x_axis, r2_train, label='R2 train')
    plt.plot(x_axis, r2_test, label='R2 test')
    plt.xlabel(x_axis_label)
    plt.ylabel('R2 Score')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()


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

    for degree in range(1, p+1):
        polynomial_degree.append(degree)

        # Extract the relevant columns from design matrix for the current degree
        X_train_sliced = X_train[:, :degree] 
        X_test_sliced = X_test[:, :degree]
        
        # OLS Regression
        theta_OLS = OLS_parameters(X_train_sliced, y_train)
        y_tilde_train = X_train_sliced @ theta_OLS
        y_tilde_test = X_test_sliced @ theta_OLS

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


        # Sklearn Linear Regression without intercept for validation of code, test dataset only.
        # only for validation of own code        
        model = LinearRegression(fit_intercept=use_intercept)
        model.fit(X_train_sliced, y_train)
        y_pred_sklearn = model.predict(X_test_sliced)
        mse_sklearn = MSE(y_test, y_pred_sklearn)
        r2_sklearn = R2(y_test, y_pred_sklearn)

        if verbose:
            print(f"Polynomial degree: {degree}, Sklearn test R2: {r2_sklearn}, Sklearn test MSE: {mse_sklearn}")
            print(f"Polynomial degree: {degree}, R2 test: Own - sklearn {r2_test_OLS - r2_sklearn}, MSE test: Own - sklearn {mse_test_OLS - mse_sklearn}")
            print('\n') # just to add line shift between different degrees in output
        
    return polynomial_degree, mse_train, mse_test, r2_train, r2_test


def explore_lambda(X_train, X_test, y_train, y_test, lambd,n=50,verbose=False):
    """
    Explores the effect of polynomial degree on MSE and R2 for
    both training and test datasets using OLS regression.

    Returns
    -------

    lambdas: list
        list of lambda values explored
    
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
    
    lambd : tuple (int,int)
        assuming lambdas will be generated using 
        np.logspace
    
    n : int
        number of generated values for lambda, default set to 50
    """

    lambdas = []
    mse_train = []
    mse_test = []
    r2_train  = []
    r2_test = []

    
    for l in np.logspace(lambd[0],lambd[1],n):
        lambdas.append(l)

        # Apply ridge regression
        theta_ridge = Ridge_parameters(X_train, y_train,l)
        y_tilde_train = X_train @ theta_ridge
        y_tilde_test = X_test @ theta_ridge

        # Calculate MSE for training and test data
        mse_train_ridge = MSE(y_train, y_tilde_train)
        mse_test_ridge = MSE(y_test, y_tilde_test)
        mse_train.append(mse_train_ridge)
        mse_test.append(mse_test_ridge)
        if verbose: print(f"Lambda: {l}, MSE_train_ridge: {mse_train_ridge}, MSE_test_OLS: {mse_test_ridge}")

        # Calculate R2 for training and test data
        r2_train_ridge = R2(y_train, y_tilde_train)
        r2_test_ridge = R2(y_test, y_tilde_test)
        r2_train.append(r2_train_ridge)
        r2_test.append(r2_test_ridge)
        if verbose: print(f"Lambda: {l}, R2_train_ridge: {r2_train_ridge}, R2_test_ridge: {r2_test_ridge}")


    
    return lambdas, mse_train, mse_test, r2_train, r2_test
