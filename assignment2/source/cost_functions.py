# COST FUNCTIONS

import autograd.numpy as np 

#parent class
class cost_function:
    # should be overwritten
    def cost(self, y_true,y_pred):
        raise NotImplementedError

    def cost_derivative(self,y_pred,y_true):
        raise NotImplementedError
    
    # overwritten if needed
    def reset(self):
        pass

class mse(cost_function):

    def cost(self,y_true, y_pred):
        diff = (y_pred - y_true)
        return np.mean(diff ** 2)
    
    def cost_derivative(self, y_pred, y_true):
        B, K = y_pred.shape  
        return 2.0 * (y_pred - y_true) / (B * K)
    
class cross_entropy(cost_function):  

    def cost(self,y_true, y_pred, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def cost_derivative(self,y_true, y_pred):
        return y_pred - y_true

    def CostCrossEntropy(self,target):
        
        def func(X):
            return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

        return func


