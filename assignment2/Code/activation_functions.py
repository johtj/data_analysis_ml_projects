# ACTIVATION FUNCTIONS

import autograd.numpy as np 


def sigmoid(Z): # binary classification
    # ref 1
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)


def RELU(Z):
    return np.where(Z > 0, Z, 0)

    

def RELU_derivative(Z):
    return np.where(Z > 0, 1, 0)




def LRELU(Z, delta = 10e-4):
    # ref 1
    return np.where(X > np.zeros(Z.shape), Z, delta * Z)

def LRELU_derivative(Z, delta = 10e-4):
    return np.where(Z > np.zeros(Z.shape), 1, delta)




def linear(Z):
    return Z

def linear_derivative(Z):
    return np.ones_like(Z)


def softmax(Z): #multiclass classification
    Z = np.asarray(Z)
    Z_shift = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / expZ.sum(axis=1, keepdims=True)


