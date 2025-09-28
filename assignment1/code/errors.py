import numpy as np

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


def variance(y_predict):
    """"
    Calculates the variance of the predicted values

    Returns
    -------

    variance: float
        Variance value
    
    Parameters
    ----------

    y_predict: numpy array shape (n)
        Predicted y values of the data set.
    """

    variance = np.mean(np.var(y_predict, axis=1))
    return variance

def squared_bias(y, y_predict):
    """"
    Calculates the squared bias between true and predicted values. Squaring bias so it is better for plotting.

    Returns
    -------

    squared_bias: float
        Bias value
    
    Parameters
    ----------
    y : numpy array shape (n)
    Y values of the data set. 
    
    y_predict: numpy array shape (n)
        Predicted y values of the data set.
    """

    squared_bias = np.mean((y - np.mean(y_predict, axis=1))**2)
    return squared_bias