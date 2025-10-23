import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return y_pred - y_true



def cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true



def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func




def cost(layers, input, activation_funcs, target):

    """
    Computes the cost (error) between the predicted output of a feedforward neural network
    and the target output.

    Parameters:
    ----------
    layers : list of tuples
        A list of (W, b) tuples representing the weights and biases for each layer.
    
    input : np.ndarray
        The input vector to the network, typically of shape (input_size,).
    
    activation_funcs : list of callable
        A list of activation functions to apply after each layer's linear transformation.
    
    target : np.ndarray
        The expected output vector (ground truth) for the given input.

    Returns:
    -------
    float
        The cost value, typically computed as the mean squared error (MSE) between
        the predicted output and the target.
    """

    predict = feed_forward(input, layers, activation_funcs)
    return mse(predict, target)