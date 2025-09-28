import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from main_methods import OLS_parameters,Ridge_parameters
from errors import MSE,R2, squared_bias, variance

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


def plot_bias_variance_tradeoff_polynomial_degree_sklearn(x, y, p=65, bootstraps=200):

    x_train, x_test, y_train, y_test = train_test_split(
        x[:, None], y, random_state=1, test_size=0.2
    )

    degrees = np.arange(1, p + 1, step=5)

    mses = np.zeros(degrees.shape)
    variances = np.zeros(degrees.shape)
    biases = np.zeros(degrees.shape)

    for i, degree in enumerate(degrees):
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression(fit_intercept=True)
        )

        preds = []  # will hold predictions on x_test for each bootstrap (shape per item: (n_test,))
        for j in range(bootstraps):
            x_train_re, y_train_re = resample(
                x_train, y_train,
                random_state=j
            )

            model.fit(x_train_re, y_train_re)
            preds.append(model.predict(x_test))  # (n_test,)

        # Stack to shape (n_test, bootstraps)
        preds = np.column_stack(preds)

        mses[i] = MSE(y_test[:, None], preds)
        variances[i] = variance(preds)
        biases[i] = squared_bias(y_test, preds)

    plt.plot(degrees, mses, label="MSE (test)")
    plt.plot(degrees, variances, label="Variance (test)")
    plt.plot(degrees, biases, label="Bias^2 (test)")
    plt.legend()
    plt.yscale("log")
    plt.show()

def plot_bias_variance_tradeoff_datapoints(x, y, max_n=500, degree=20, bootstraps = 200):
    """
    Plots MSE, bias and variance for different numbers of datapoints (bias-variance tradeoff)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x[:, None], y, random_state=1, test_size=0.2
    )

    n_test = np.linspace(10, max_n, 10).astype(int)

    mses = np.zeros(n_test.shape)
    variances = np.zeros(n_test.shape)
    biases = np.zeros(n_test.shape)

    for i, n in enumerate(n_test): # loop through the data points
        # initialize predictions
        predictions = np.empty((y_test.shape[0], bootstraps))

        # Combine x transformation and model 
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=True))

        for b in range(bootstraps):
            X_train_re, y_train_re = resample(x_train, y_train, n_samples=n) # bootstrap resampling, taking 

            # fit your model on the sampled data
            # make predictions on the test data
            predictions[:,b] = model.fit(X_train_re, y_train_re).predict(x_test) # Evaluate the new model on the same test data each time.

        
        biases[i] = squared_bias(y_test, predictions)
        variances[i] = variance(predictions)
        mses[i] = MSE(y_test[:, None], predictions)

    plt.plot(n_test, mses, label="MSE (test)")
    plt.plot(n_test, variances, label="Variance (test)")
    plt.plot(n_test, biases, label="Bias^2 (test)")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Number of datapoints")
    plt.show()
