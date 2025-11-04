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





class class_mse_loss(cost_function):

    def mse_with_regularization(y_true, y_pred, weights, l1=0.0, l2=0.0):
        mse_loss = ((y_true - y_pred) ** 2).mean()
        l1_penalty = l1 * np.sum(np.abs(weights))
        l2_penalty = l2 * np.sum(weights ** 2)
        return mse_loss + l1_penalty + l2_penalty

    def mse_derivative_with_regularization(y_true, y_pred, weights, l1=0.0, l2=0.0):
        grad = 2 * (y_pred - y_true) / y_true.size
        l1_grad = l1 * np.sign(weights)
        l2_grad = 2 * l2 * weights
        return grad, l1_grad + l2_grad

class class_binary_cros_entropy(cost_function):


    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15  # to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def binary_cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)


    def binary_cross_entropy_with_regularization(y_true, y_pred, weights, l1=0.0, l2=0.0):
        base_loss = binary_cross_entropy(y_true, y_pred)
        l1_penalty = l1 * np.sum(np.abs(weights))
        l2_penalty = l2 * np.sum(weights ** 2)
        return base_loss + l1_penalty + l2_penalty

    def binary_cross_entropy_derivative_with_regularization(y_true, y_pred, weights, l1=0.0, l2=0.0):
        base_grad = binary_cross_entropy_derivative(y_true, y_pred)
        l1_grad = l1 * np.sign(weights)
        l2_grad = 2 * l2 * weights
        return base_grad, l1_grad + l2_grad


class class_multiclass_cross_entropy(cost_function):

    def softmax(logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def multiclass_cross_entropy(y_true, logits):
        probs = softmax(logits)
        return -np.mean(np.sum(y_true * np.log(probs + 1e-15), axis=1))

    def multiclass_cross_entropy_derivative(y_true, logits):
        probs = softmax(logits)
        return (probs - y_true) / y_true.shape[0]
    
    def multiclass_cross_entropy_with_regularization(y_true, logits, weights, l1=0.0, l2=0.0):
        base_loss = multiclass_cross_entropy(y_true, logits)
        l1_penalty = l1 * np.sum(np.abs(weights))
        l2_penalty = l2 * np.sum(weights ** 2)
        return base_loss + l1_penalty + l2_penalty

    def multiclass_cross_entropy_derivative_with_regularization(y_true, logits, weights, l1=0.0, l2=0.0):
        base_grad = multiclass_cross_entropy_derivative(y_true, logits)
        l1_grad = l1 * np.sign(weights)
        l2_grad = 2 * l2 * weights
        return base_grad, l1_grad + l2_grad

