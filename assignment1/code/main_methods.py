import numpy as np


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


def Ridge_parameters(X, y, lamb):
    """
        Calculates the optimal parameters, r_params, using the 
        ridge regression method.  

        r_params = inv(X.T @ X + lambda I) @ X.T @ y

        Returns
        -------
        r_params : numpy array shape (n)
            the optimal parameters, theta as given by the
            Ridge regression method. 

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.
        
        y : numpy array shape (n)
            Y values of the data set.     
    """
        
    # Assumes X is scaled and has no intercept column    
    
    p = X.shape[1]
    I = np.eye(p)

    r_params = np.linalg.inv(X.T @ X + lamb * I) @ X.T @ y

    return r_params

def gradient_descent_ridge(X,y,eta,lam,num_iters,n_features):
    """
        Calculates the optimal parameters, theta, using the 
        ridge regression and gradient descent

        Returns
        -------
        theta_gdRidge : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method.

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.

        y : numpy array shape (n)
            Y values of the data set. 
        
        eta : int
            gradient descent parameter
    
        lam : int
            learning rate
        
        num_iters : int
            number of iterations

        n_features : int
            number of features in feature matrix
    """

    # Initialize weights for gradient descent
    theta_gdRidge = np.zeros(n_features)

    # Gradient descent loop
    for t in range(num_iters):
        # Compute gradients for Ridge
        grad_Ridge = (2.0/n) * X.T @(X @ theta_gdRidge - y) + 2*lam*theta_gdRidge

        # Update parameters theta
        theta_gdRidge -= eta*grad_Ridge 

    # After the loop, theta contains the fitted coefficients
    return theta_gdRidge



def gradient_descent_OLS(X,y,eta,num_iters,n_features):
    """
        Calculates the optimal parameters, theta, using the 
        ordinary least squares method and gradient descent

        Returns
        -------
        theta_gdOLS : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method.

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.

        y : numpy array shape (n)
            Y values of the data set. 
        
        eta : int
            gradient descent parameter
    
        lam : int
            regularization
        
        num_iters : int
            number of iterations

        n_features : int
            number of features in feature matrix
    """
    
    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(n_features)

    # Gradient descent loop
    for t in range(num_iters):
        # Compute gradients for OSL and Ridge
        grad_OLS = (2.0/n)*X.T @ (X @ theta_gdOLS - y)

        # Update parameters theta
        theta_gdOLS -= eta*grad_OLS

    # After the loop, theta contains the fitted coefficients
    return theta_gdOLS


## Implementation of Lasso regression with gradient descent

def soft_threshold(rho, alpha):
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0.0

def lasso_gradient_descent(X, y, lambda_, learning_rate, tol, max_iter, fit_intercept):

    """
    Calculates the optimal parameters and intercept using the 
    Lasso method with gradient descent

    Code retreived from Microsoft Copilot 25.09.2025
    Question 1: I need a native python code, not using sklearn or similar, for lasso regression implemented as gradient descent. Tolerance criteria and max iteration parameter must be implemented. 
    It must be an parameter to choose to calculate intercept or not. Soft thresolding must be used to converge coeficients to 0 if needed. 
    Input feature matrix X is already scaled. Compare the provided code with sklearn to confirm that the two methods give the same result
    Question 2: Can you write code as a function?
    Question 3: The lasso regression and code should contain both gradient and lambda parameters?
    Question 4: How to scale intercept back to original values, it is centered in code

    Returns
    -------
    coef: numpy array shape (n)
        the optimal parameters, theta as given by the
        Lasso method.

    intercept: float
        intercept value from regression
        
    Parameters
    ----------
    X : numpy array shape (n,f)
        Feature matrix for the data, where n is the number
        of data points and f is the number of features.

    y : numpy array shape (n)
        Y values of the data set. 
    
    lambda_ : int
        regularization
    
    learning_rate : float
        gradient descent parameter

    tol: float
        tolerance for convergence stopping criteria
    
    max_iter : int
        number of iterations

    fit_intercept : Bool
        Bool to determine if intercept should be included or not in regression:
        False : no intercept 
        True : include intercept
    """

    n_samples, n_features = X.shape
    coef = np.zeros(n_features)

    if fit_intercept:
        y_mean = np.mean(y)
        y_centered = y - y_mean
    else:
        y_centered = y

    for iteration in range(max_iter):
        coef_old = coef.copy()
        gradient = -2 * X.T @ (y_centered - X @ coef) / n_samples
        coef -= learning_rate * gradient

        for j in range(n_features):
            coef[j] = soft_threshold(coef[j], learning_rate * lambda_)

        if np.sum(np.abs(coef - coef_old)) < tol:
            break

    if fit_intercept:
        intercept = y_mean - np.mean(X, axis=0) @ coef
    else:
        intercept = 0.0

    return coef, intercept


def sklearn_lasso_regression(X, y, alpha, use_intercept, max_iterations, tolerance, verbose=False):
    """
    Calculates the optimal parameters and intercept using the 
    Lasso method with Sklearn. Sklearn uses coordiante descent and 
    not gradient descent. Used to compare and verify own code.

    Returns
    -------
    None
        
    Parameters
    ----------
    X : numpy array shape (n,f)
        Feature matrix for the data, where n is the number
        of data points and f is the number of features.

    y : numpy array shape (n)
        Y values of the data set. 

    use_intercept : Bool
        Bool to determine if intercept should be included or not in regression:
        False : no intercept 
        True : include intercept
    
    max_iterations : int
        number of iterations
            
    tolerance: float
        tolerance for convergence stopping criteria
            
    verbose : Bool
        Include verbose output from function, default set to false
    """
    from sklearn.linear_model import Lasso    
    RegLasso = Lasso(alpha, fit_intercept=use_intercept, max_iter=max_iterations, tol=tolerance)
    RegLasso.fit(X, y)
    if verbose: print(f"Sklearn coef: {RegLasso.coef_}, Sklearn intercept: {RegLasso.intercept_}")