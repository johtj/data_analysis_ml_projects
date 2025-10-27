# ACTIVATION FUNCTIONS

import autograd.numpy as np 

# 1- Lecture notes FYS-STK4155 - week 42
#     https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/week42.html#id3


def sigmoid(X):
    # ref 1
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    s = sigmoid(X)
    return s * (1 - s)


def RELU(X):
    # ref 1
    #return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))
    return np.where(X > 0, X, 0)

    

def RELU_derivative(X):
    #return (X > np.zeros(X.shape) > 0).astype(float)
    return np.where(X > 0, 1, 0)




def LRELU(X, delta = 10e-4):
    # ref 1
    return np.where(X > np.zeros(X.shape), X, delta * X)

def LRELU_derivative(X, delta = 10e-4):
    return np.where(X > np.zeros(X.shape), 1, delta)




def linear(X):
    return X

def linear_derivative(X):
    return np.ones_like(X)


