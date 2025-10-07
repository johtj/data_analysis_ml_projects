import numpy as np

def gradient_descent_OLS(X,y,eta,num_iters):
    """
        Calculates the optimal parameters, theta, using the 
        ordinary least squares method and gradient descent

        Returns
        -------
        theta_gdOLS : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method.

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.

        y : numpy array shape (n)
            Y values of the data set. 
        
        eta : int
            learning rate
        
        num_iters : int
            number of iterations

        
    """
    
    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(np.shape(X)[1])
    n = X.shape[0]

    # Gradient descent loop
    for t in range(num_iters):
        # Compute gradients for OSL and Ridge
        grad_OLS = (2.0/n)*X.T @ (X @ theta_gdOLS - y)

        # Update parameters theta
        theta_gdOLS -= eta*grad_OLS

    # After the loop, theta contains the fitted coefficients
    return theta_gdOLS


def gradient_descent_OLS_momentum(X,y,eta,num_iters,momentum):
    """
        Calculates the optimal parameters, theta, using the 
        ordinary least squares method and gradient descent with momentum

        Returns
        -------
        theta_gdOLS : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method.

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.

        y : numpy array shape (n)
            Y values of the data set. 
        
        eta : int
            learning rate
        
        num_iters : int
            number of iterations
        
        momentum: float
            momentum weight

    """
    
    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(np.shape(X)[1])
    change = 0.0
    n = X.shape[0]

    # Gradient descent loop
    for t in range(num_iters):
        # Compute gradients for OSL and Ridge
        grad_OLS = (2.0/n)*X.T @ (X @ theta_gdOLS - y)

        update = eta * grad_OLS + momentum*change

        # Update parameters theta
        theta_gdOLS -= eta*grad_OLS

        #update change 
        change = update

    # After the loop, theta contains the fitted coefficients
    return theta_gdOLS

def ADAgrad_OLS(X,y,eta,num_iters):
    """
        Calculates the optimal parameters, theta, using the 
        ordinary least squares method and gradient descent with 
        adaptive learning rate using ADAgrad_OLS

        Returns
        -------
        theta_gdOLS : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method.

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.

        y : numpy array shape (n)
            Y values of the data set. 
        
        eta : int
            learning rate
        
        num_iters : int
            number of iterations

    """
    
    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(np.shape(X)[1]) #initial theta
    n = X.shape[0]
    r = 0 #gradient accumulation variable
    delta = 10e-7
    # Gradient descent loop

    for t in range(num_iters):

        # Compute gradients for OSL
        grad_OLS = (2.0/n)*X.T @ (X @ theta_gdOLS - y)

        #accumulate squared gradient 
        r = r + (grad_OLS * grad_OLS)

        #compute update
        update = (eta / (delta + np.sqrt(r))) * grad_OLS

        # Update parameters theta
        theta_gdOLS -= update

    return theta_gdOLS

def RMSprop_OLS(X,y,eta,num_iters):
    '''
        Calculates the optimal parameters, theta, using the 
        ordinary least squares method and gradient descent with 
        adaptive learning rate using RMSprop_OLS
   
        Returns
        -------
        theta_gdOLS : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method.

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.

        y : numpy array shape (n)
            Y values of the data set. 
        
        eta : int
            learning rate
        
        num_iters : int
            number of iterations
      
    '''
    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(np.shape(X)[1]) #initial theta
    n = X.shape[0]
    decay_rate = 0
    r = 0 #gradient accumulation variable
    delta = 10e-6 #stabilize division by small numbers
    # Gradient descent loop

    for t in range(num_iters):

        # Compute gradients for OSL
        grad_OLS = (2.0/n)*X.T @ (X @ theta_gdOLS - y)

        #accumulate squared gradient 
        r = decay_rate * r + (1-decay_rate) * (grad_OLS * grad_OLS)

        #compute update
        update = (eta / (delta + np.sqrt(r))) * grad_OLS

        # Update parameters theta
        theta_gdOLS += update

    return theta_gdOLS

def ADAM_OLS(X,y,eta,num_iters):
    '''
        Calculates the optimal parameters, theta, using the 
        ordinary least squares method and gradient descent with 
        adaptive learning rate using ADAM
   
        Returns
        -------
        theta_gdOLS : numpy array shape (n)
            the optimal parameters, theta as given by the
            OLS method.

        Parameters
        ----------
        X : numpy array shape (n,f)
            Feature matrix for the data, where n is the number
            of data points and f is the number of features.

        y : numpy array shape (n)
            Y values of the data set. 
        
        eta : int
            learning rate
        
        num_iters : int
            number of iterations
    '''
    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(np.shape(X)[1]) #initial theta
    n = X.shape[0]
    decay1= 0.9
    decay2 = 0.999
    s = 0
    r = 0 #1st & 2nd moment variables 
    ts = 0 # time step
    delta =10e-8 #numerical stabilization

    # Gradient descent loop

    for t in range(num_iters):

        # Compute gradients for OSL
        grad_OLS = (2.0/n)*X.T @ (X @ theta_gdOLS - y)
        ts += 1

        s = decay1 * s  + (1-decay1)*grad_OLS  
        r = decay2*r + (1-decay2)*grad_OLS*grad_OLS

        s_debias = s / (1-decay1**ts)
        r_debias = r / (1-decay2**ts)

        update = -eta*(s_debias/np.sqrt(r_debias)+delta)
        
        # Update parameters theta
        theta_gdOLS += update

    return theta_gdOLS