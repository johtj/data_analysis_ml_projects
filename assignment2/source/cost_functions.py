# COST FUNCTIONS

import autograd.numpy as np 

class cost_functions:
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



