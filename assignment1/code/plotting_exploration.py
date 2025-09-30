import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from main_methods import OLS_parameters,Ridge_parameters, lasso_gradient_descent
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
            print(f"Polynomial degree: {degree}, Coef: {model.coef_}, intercept: {model.intercept_}")
            print('\n') # just to add line shift between different degrees in output
    
    return polynomial_degree, mse_train, mse_test, r2_train, r2_test, thetas


def explore_lambda(X_train, X_test, y_train, y_test, lambdas, verbose=False):
    """
    Explores the effect of polynomial degree on MSE and R2 for
    both training and test datasets using Ridge regression.

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
    
    lambdas : numpy array shape (n)
        lambda values
    
    verbose : Bool
        Include verbose output from function, default set to false
   
    """

    lambdas_list = []
    mse_train = []
    mse_test = []
    r2_train  = []
    r2_test = []

    for l in lambdas:
        lambdas_list.append(l)

        # Apply ridge regression
        theta_ridge = Ridge_parameters(X_train, y_train,l)
        y_tilde_train = X_train @ theta_ridge
        y_tilde_test = X_test @ theta_ridge
        if verbose: print(f"Ridge: Lambda: {l}, Coef: {theta_ridge}")


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
        if verbose: print(f"Lambda: {l}, R2_train_ridge: {r2_train_ridge}, R2_test_ridge: {r2_test_ridge} \n")

    return lambdas_list, mse_train, mse_test, r2_train, r2_test, theta_ridge



def lasso_grid_search(X_train, X_test, y_train, y_test, lambdas, learning_rate, tol, max_iter, fit_intercept, verbose=False):
    """
    Grid search og lamda and eta values for finding optimal parameters for Lasso
    regression, where lowest mse is defined as optimal criterion

    Returns
    -------
    best_lasso: list
        list with dictionaries for lambda, learning rate, mse, coef and intercept for all
        analyzed variants of lambda and eta values
    
    mse_train: float
        value for MSE based on training data
    
    mse_test: float
        value for MSE based on test data
    
    r2_train: float
        value for R2 based on training data
    
    r2_test: float
        value for R2 based on test data
    
    mse_values: list
        list with mse values for combinations of lambda and eta
        
    Parameters
    ----------
    X_train : numpy array shape (n,f)
        Feature matrix for the train data, where n is the number
        of data points and f is the number of features.
    
    X_test : numpy array shape (n,f)
        Feature matrix for the test data, where n is the number
        of data points and f is the number of features.

    y_train : numpy array shape (n)
        Y values of the train data set. 

    y_test : numpy array shape (n)
        Y values of the test data set. 
    
    lambda_ : int
        regularization
    
    learning_rate : float
        gradient descent parameter
    
    tol: float
        tolerance for convergence stopping criteria
    
    max_iter : int
        number of iterations

    use_intercept : Bool
        Bool to determine if intercept should be included or not in regression:
        False : no intercept 
        True : include intercept
        
    verbose : Bool
        Include verbose output from function, default set to false
    """

    lasso_results = []
    mse_train = []
    mse_test = []
    r2_train = []
    r2_test = []

    for lambda_ in lambdas:
        for lr in learning_rate:
            coef, intercept = lasso_gradient_descent(X_train, y_train, lambda_, lr, tol, max_iter, fit_intercept)
            y_tilde_test = X_test @ coef  + intercept
            mse = MSE(y_test, y_tilde_test)
            lasso_results.append({
                'lambda': lambda_,
                'learning_rate': lr,
                'mse': mse,
                'coef': coef,
                'intercept': intercept
            })

    best_lasso = min(lasso_results, key=lambda r: r['mse'])

    
    # Extract all MSE values from the list
    mse_values = np.array([entry['mse'] for entry in lasso_results if 'mse' in entry])


    y_tilde_train = X_train @ best_lasso['coef'] + best_lasso['intercept']
    y_tilde_test = X_test @ best_lasso['coef'] + best_lasso['intercept']

    # Calculate MSE for training and test data
    mse_train_lasso = MSE(y_train, y_tilde_train)
    mse_test_lasso = MSE(y_test, y_tilde_test)
    mse_train.append(mse_train_lasso)
    mse_test.append(mse_test_lasso)
    if verbose: print(f"Lasso: Lambda: {lambda_}, MSE_train_lasso: {mse_train_lasso}, MSE_test_lasso: {mse_test_lasso}")
    
    # Calculate R2 for training and test data
    r2_train_lasso = R2(y_train, y_tilde_train)
    r2_test_lasso = R2(y_test, y_tilde_test)
    r2_train.append(r2_train_lasso)
    r2_test.append(r2_test_lasso)
    if verbose: print(f"Lasso: Lambda: {lambda_}, R2_train_lasso: {r2_train_lasso}, R2_test_lsso: {r2_test_lasso}")

    if verbose:
        print(f"Lasso own implementation best result: Lambda: {best_lasso['lambda']}, Learning rate: {best_lasso['learning_rate']}, MSE: {best_lasso['mse']}")
        print(f"Lasso coef: {best_lasso['coef']}, intercept: {best_lasso['intercept']}")

    return best_lasso, mse_train, mse_test, r2_train, r2_test, mse_values




def plot_heatmap_lasso(mse_array, lambda_n, etas):
    """
    Plotting of heatmap from mse values from lasso_grid_search
    Depends on number of lambda values to explore and learning rate (etas)

    Returns
    -------
    None
        
    Parameters
    ----------
    mse_array : numpy array shape (n)
        array with mse values 

    lambdas_n : int
        number of lamda values explored, regularization parameter

    etas : list
        eta (learning rate) values explored
    """

    mse_matrix = np.array(mse_array).reshape((lambda_n, len(etas)))
    print(mse_matrix.shape)


    plt.figure(figsize=(6,4))
    fig, ax = plt.subplots()
    im = ax.imshow(mse_matrix)

    for i in range(0, (lambda_n)):
        for j in range(len(etas)):
            text = ax.text(j, i, f"{mse_matrix[i, j]:.2e}",
                        ha="center", va="center", color="w")
            
    ax.set_ylabel("Lambdas", fontsize=12)
    ax.set_xlabel("learning rate", fontsize=12)
    ax.set_title("Mean Squared Error (MSE) for different lambda values and learning rate", fontsize=12)
    plt.colorbar(im)
    plt.show()


def plot_theta_by_polynomials(thetas, degree):
    """
    Plotting thetas as function of polynomial degree

    Returns
    -------
    None
        
    Parameters
    ----------
    thetas: list
        with values of theta for each polynomial degree analysed

    degree : int
        Polynomial degree used in analysis
    """
    for i, theta in enumerate(thetas):
        plt.plot(range(len(theta)), theta, label=f'Polynomial degree {i + 1}')
    
    plt.xlabel('Degree')
    plt.ylabel('Theta Value')
    plt.title('Theta values by polynomial degree {degree} - OLS regression')
    
    # Place legend outside the plot area to the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend
    plt.savefig(f'Theta values by polynomial degree {degree} - OLS regression')
    plt.show()
    plt.close()






def plot_xy_xynoise_ypredicted(x, y, x_train, y_train, y_predicted_rescaled, x_test, n_datapoints, regression_method, poly_degree, noise):
    """
    Plot x, y, x_train, y_train and x_test with predicted y_values at original scale.
    Plots are shown with variables as n_datapoints, regression_method, poly_degree and noise

    Returns
    -------
    Saved figure
        
    Parameters
    ----------
    x : numpy array shape (n)
        x values of function without noise
    
    y : numpy array shape (n)
        y values of function without noise
    
    x_noise : numpy array shape (n)
        x values of function with noise
    
    y_noise : numpy array shape (n)
        y values of function with noise

    y_predicted_rescaled : numpy array shape (n)
        Predicted y values at original scale

    x_test : numpy array shape (n)
        x values of test dataset

    n_datapoints : int
        variable used for plotting of n_datapoints in regression

    regression_method : string
        String for regression method in plot
    
    n_datapoints : int
        variable used for plotting of n_datapoints in regression

    poly_degree : int
        variable used for plotting of poly_degree in regression

    noise : bool
        decide if noise or without noise should be in the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(x,y, marker='o', color='blue', label='Runges function - no noise')
    plt.scatter(x_train, y_train, marker='o', color='green', label='Runges function - training data')
    plt.scatter(x_test, y_predicted_rescaled, marker='o', color='red', label='Runges function - predicted and rescaled')
    plt.legend()
    if noise:
        text = f'Runges function with {regression_method} regression with noise\nNumber of data points: {n_datapoints} Polynomial degree: {poly_degree}'
        filename = f'Runges function with {regression_method} regression with noise - Number of data points: {n_datapoints} Polynomial degree - {poly_degree}.png'
        plt.title(text)
    else:
        text = f'Runges function with {regression_method} regression without noise\nNumber of data points: {n_datapoints} Polynomial degree: {poly_degree}'
        filename = f'Runges function with {regression_method} regression without noise - Number of data points: {n_datapoints} Polynomial degree - {poly_degree}.png'
        plt.title(text)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.savefig(filename)
    plt.show()
    plt.close()