
import numpy as np

def grad_Ridge(n,X,y,theta_gdRidge,lam):
    
    return (2.0/n) * X.T @(X @ theta_gdRidge - y) + 2*lam*theta_gdRidge

def grad_OLS(n,X,y,theta_gdOLS):
    return (2.0/n)*X.T @ (X @ theta_gdOLS - y)

def ADAM(grads,eta,timestep,moment,second,decay1=0.9,decay2=0.999,delta=10e-8):

    moment = decay1 * moment  + (1-decay1)*grads  
    second = decay2* second + (1-decay2)*grads*grads

    moment_debias = moment / (1-decay1**timestep)
    second_debias = second / (1-decay2**timestep)

    change = -eta*(moment_debias/np.sqrt(second_debias)+delta)
            
    return change, moment,second


def RMSprop(grads, eta, r, decay_rate=0.9,delta=10e-6):
    #accumulate squared gradient 
    r = decay_rate * r + (1-decay_rate) * (grads * grads)

    #compute update
    change = (eta / (delta + np.sqrt(r))) * grads

    return change

