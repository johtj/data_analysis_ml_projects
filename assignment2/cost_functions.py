# COST FUNCTIONS

import autograd.numpy as np 
from activation_functions import softmax


"""
old - just kept to pass information
def mse(y_true, y_pred):
    diff = (y_pred - y_true)
    return np.mean(diff ** 2)



def mse_derivative(y_pred, y_true):
    B, K = y_pred.shape  
    return 2.0 * (y_pred - y_true) / (B * K)



def cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true



def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func
"""


def mse_loss_basic(y_pred, y_true, *, l2=0.0, l1=0.0):
    """
    Added L1 and l2 term with Copilot
    Basic MSE loss with optional L1 and L2 regularization on predictions.

    Parameters
    ----------
    y_pred : ndarray, shape (n,)
        Predicted values.
    y_true : ndarray, shape (n,)
        True target values.
    l2 : float, default 0.0
        L2 regularization strength.
    l1 : float, default 0.0
        L1 regularization strength.

    Returns
    -------
    loss : float
        Total loss including regularization.
    """
    residual = y_pred - y_true
    data_loss = np.mean(residual**2)

    reg_loss = 0.0
    if l2 != 0.0:
        reg_loss += l2 * np.dot(y_pred, y_pred)
    if l1 != 0.0:
        reg_loss += l1 * np.sum(np.abs(y_pred))

    return data_loss + reg_loss

def mse_loss_gradient(y_pred, y_true, *, l2=0.0, l1=0.0):
    """
    Added L1 and l2 term with Copilot
    Gradient of MSE loss with optional L1 and L2 regularization.

    Parameters
    ----------
    y_pred : ndarray, shape (n,)
    y_true : ndarray, shape (n,)
    l2 : float
        L2 regularization strength.
    l1 : float
        L1 regularization strength.

    Returns
    -------
    grad : ndarray, shape (n,)
        Gradient of the loss with respect to y_pred.
    """
    grad = 2 * (y_pred - y_true) / y_pred.size

    if l2 != 0.0:
        grad += 2 * l2 * y_pred
    if l1 != 0.0:
        grad += l1 * np.sign(y_pred)

    return grad



def binary_cross_entropy_regularized(y_pred, y_true, *, l1=0.0, l2=0.0, eps=1e-15):
    """
    Added L1 and l2 term with Copilot
    Binary cross-entropy loss with optional L1 and L2 regularization on predictions.

    Parameters
    ----------
    y_pred : ndarray, shape (n,)
        Predicted probabilities (between 0 and 1).
    y_true : ndarray, shape (n,)
        True binary labels (0 or 1).
    l1 : float
        L1 regularization strength.
    l2 : float
        L2 regularization strength.
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    loss : float
        Total loss including regularization.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    data_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    reg_loss = 0.0
    if l2 != 0.0:
        reg_loss += l2 * np.sum(y_pred**2)
    if l1 != 0.0:
        reg_loss += l1 * np.sum(np.abs(y_pred))

    return data_loss + reg_loss


def binary_cross_entropy_gradient(y_pred, y_true, eps=1e-15):
    """
    Added L1 and l2 term with Copilot
    Gradient of binary cross-entropy loss with respect to y_pred.

    Parameters
    ----------
    y_pred : ndarray, shape (n,)
    y_true : ndarray, shape (n,)
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    grad : ndarray, shape (n,)
        Gradient of the loss with respect to y_pred.
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    grad = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
    grad /= y_pred.size
    return grad



def cross_entropy_gradient(logits, targets):
    """
    Optimized with Copilot
    Computes the gradient of the combined softmax activation and cross-entropy loss
    with respect to the input logits.

    Parameters
    ----------
    logits : ndarray, shape (n_samples, n_classes)
        Raw output scores (logits) from the model before applying softmax.
    targets : ndarray, shape (n_samples, n_classes)
        One-hot encoded true labels for each sample.

    Returns
    -------
    grad : ndarray, shape (n_samples, n_classes)
        The gradient of the loss with respect to the logits.

    Notes
    -----
    Exercise week 42
        IMPORTANT: Do not implement the derivative terms for softmax and cross-entropy separately,
        as it is complex and error-prone. Instead, use the fact that their combination simplifies to:
            gradient = prediction - target

    This simplification is well-known and widely used in deep learning frameworks.
    See:
    - https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    """
    pred_probs = softmax(logits)
    grad = pred_probs - targets
    return grad

