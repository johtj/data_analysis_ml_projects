import autograd.numpy as np

class activation_function:
    # should be overwritten
    def activation_function(self, y_true,y_pred):
        raise NotImplementedError

    def activation_derivative(self,y_pred,y_true):
        raise NotImplementedError
    
    # overwritten if needed
    def reset(self):
        pass


class sigmoid(activation_function):

    def sigmoid(self,X):
        # ref 1
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self,X):
        s = self.sigmoid(X)
        return s * (1 - s)

class RELU(activation_function):

    def RELU(self,X):
        # ref 1
        #return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))
        return np.where(X > 0, X, 0)

    def RELU_derivative(self,X):
        #return (X > np.zeros(X.shape) > 0).astype(float)
        return np.where(X > 0, 1, 0)

class LRELU(activation_function):
    def LRELU(self,X, delta = 10e-4):
        # ref 1
        return np.where(X > np.zeros(X.shape), X, delta * X)

    def LRELU_derivative(self,X, delta = 10e-4):
        return np.where(X > np.zeros(X.shape), 1, delta)

class linear(activation_function):

    def linear(self,X):
        return X

    def linear_derivative(self,X):
        return np.ones_like(X)