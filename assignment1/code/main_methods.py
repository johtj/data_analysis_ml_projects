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
            learning rate
        
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