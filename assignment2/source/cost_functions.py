# COST FUNCTIONS

import autograd.numpy as np 

#parent class
class cost_function:
    # should be overwritten
    @staticmethod
    def cost(y_true,y_pred):
        raise NotImplementedError

    @staticmethod
    def cost_derivative(y_pred,y_true):
        raise NotImplementedError
    

class mse(cost_function):

    @staticmethod
    def cost(y_true, y_pred):
        diff = (y_pred - y_true)
        return np.mean(diff ** 2)
    
    @staticmethod
    def cost_derivative(y_pred, y_true):
        B, K = y_pred.shape  
        return 2.0 * (y_pred - y_true) / (B * K)
    

class cross_entropy(cost_function):  

    @staticmethod
    def cost(y_true, y_pred, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def cost_derivative(y_true, y_pred):
        return y_pred - y_true

    @staticmethod
    def CostCrossEntropy(target):
        
        def func(X):
            return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

        return func

#parent class
class cost_function_regularized:
    # should be overwritten
    def __init__(self,l1: float = 0.0,l2: float = 0.0):
        self.l1 = l1
        self.l2 = l2

    def cost(self,y_true,y_pred,weights):
        raise NotImplementedError

    def cost_derivative(self,y_pred,y_true,weights):
        raise NotImplementedError

class class_mse_loss(cost_function_regularized):

    def __init__(self,l1: float = 0.0,l2: float = 0.0):
        super().__init__(l1,l2)


    def cost(self,y_true, y_pred, weights):
        mse_loss = ((y_true - y_pred) ** 2).mean()
        l1_penalty = self.l1 * np.sum(np.abs(weights))
        l2_penalty = self.l2 * np.sum(weights ** 2)
        return mse_loss + l1_penalty + l2_penalty

    def cost_derivative(self,y_true, y_pred, weights):
        grad = 2 * (y_pred - y_true) / y_true.size
        l1_grad = self.l1 * np.sign(weights)
        l2_grad = 2 * self.l2 * weights
        return grad, l1_grad + l2_grad

class class_binary_cros_entropy(cost_function_regularized):

    def __init__(self,l1: float = 0.0,l2: float = 0.0):
        super().__init__(l1,l2)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15  # to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)

    #binary_cross_entropy_with_regularization
    def cost(self,y_true, y_pred, weights):
        base_loss = self.binary_cross_entropy(y_true, y_pred)
        l1_penalty = self.l1 * np.sum(np.abs(weights))
        l2_penalty = self.l2 * np.sum(weights ** 2)
        return base_loss + l1_penalty + l2_penalty

    #binary_cross_entropy_derivative_with_regularization
    def cost_derivative(self,y_true, y_pred, weights):
        base_grad = self.binary_cross_entropy_derivative(y_true, y_pred)
        l1_grad = self.l1 * np.sign(weights)
        l2_grad = 2 * self.l2 * weights
        return base_grad, l1_grad + l2_grad


class class_multiclass_cross_entropy(cost_function_regularized):
    
    def __init__(self,l1: float = 0.0,l2: float = 0.0):
        super().__init__(l1,l2)

    @staticmethod
    def softmax(logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def multiclass_cross_entropy(self,y_true, logits):
        probs = self.softmax(logits)
        return -np.mean(np.sum(y_true * np.log(probs + 1e-15), axis=1))

    def multiclass_cross_entropy_derivative(self,y_true, logits):
        probs = self.softmax(logits)
        return (probs - y_true) / y_true.shape[0]
    
    #multiclass_cross_entropy_with_regularization
    def cost(self,y_true, logits, weights):
        base_loss = self.multiclass_cross_entropy(y_true, logits)
        l1_penalty = self.l1 * np.sum(np.abs(weights))
        l2_penalty = self.l2 * np.sum(weights ** 2)
        return base_loss + l1_penalty + l2_penalty

    #multiclass_cross_entropy_derivative_with_regularization
    def cost_derivative(self,y_true, logits, weights):
        base_grad = self.multiclass_cross_entropy_derivative(y_true, logits)
        l1_grad = self.l1 * np.sign(weights)
        l2_grad = 2 * self.l2 * weights
        return base_grad, l1_grad + l2_grad

