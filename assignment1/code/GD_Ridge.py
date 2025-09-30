import numpy as np

def gradient_descent_ridge(X,y,eta,lam,num_iters):
    """
        Calculates the optimal parameters, theta, using the 
        ridge regression and gradient descent

        Returns
        -------
        theta_gdRidge : numpy array shape (n)
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
            gradient descent parameter
    
        lam : int
            learning rate
        
        num_iters : int
            number of iterations
    """

    # Initialize weights for gradient descent
    theta_gdRidge = np.zeros(np.shape(X)[1])
    n = X.shape[0]

    # Gradient descent loop
    for t in range(num_iters):
        # Compute gradients for Ridge
        grad_Ridge = (2.0/n) * X.T @(X @ theta_gdRidge - y) + 2*lam*theta_gdRidge

        # Update parameters theta
        theta_gdRidge -= eta*grad_Ridge 

    # After the loop, theta contains the fitted coefficients
    return theta_gdRidge

def gradient_descent_ridge_momentum(X,y,eta,lam,num_iters, momentum ):
    """
        Calculates the optimal parameters, theta, using the 
        ridge regression and gradient descent

        Returns
        -------
        theta_gdRidge : numpy array shape (n)
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
            gradient descent parameter
    
        lam : int
            learning rate
        
        num_iters : int
            number of iterations

    
        """

    # Initialize weights for gradient descent
    theta_gdRidge = np.zeros(np.shape(X)[1])
    n = X.shape[0]
    change = 0.0

    # Gradient descent loop
    for t in range(num_iters):
        # Compute gradients for Ridge
        grad_Ridge = (2.0/n) * X.T @(X @ theta_gdRidge - y) + 2*lam*theta_gdRidge

        #calculate update
        update = eta * grad_Ridge + momentum * change

        # Update parameters theta
        theta_gdRidge -= update

        #update change
        change = update
    # After the loop, theta contains the fitted coefficients
    return theta_gdRidge

def ADAgrad_Ridge(X,y,eta,lam,num_iters):
    '''
    glob_eta = 0
    initial_theta = 0 
    small_constant = 10^{-7} # for numerical stability
    r = 0 #gradient accumulation variable

    while not stopping:
        sample mini batch {x(1), ... , x(m)} with correspdoning targets y(i)
        compute gradient g = OLS_grad
        accumulated squared gradient r = r + g @ g
        compute update update = (global_eta / small_constant + np.sqrt(r)) @ g
        apply theta = theta + update

    '''
    # Initialize weights for gradient descent
    theta_gdRidge = np.zeros(np.shape(X)[1]) #initial theta
    n = X.shape[0]
    r = 0 #gradient accumulation variable
    delta = 10e-7
    # Gradient descent loop

    for t in range(num_iters):

        # Compute gradients for OSL
        grad_Ridge = (2.0/n) * X.T @(X @ theta_gdRidge - y) + 2*lam*theta_gdRidge

        #accumulate squared gradient 
        r = r + (grad_Ridge * grad_Ridge)

        #compute update
        update = (eta / (delta + np.sqrt(r))) * grad_Ridge

        # Update parameters theta
        theta_gdRidge -= update

    return theta_gdRidge

def RMSprop_Ridge(X,y,eta,num_iters):
    '''
    global_eta = 0
    decay_rate = 0
    initial_teta = 0 
    small_cosntant = 10^{-6} #used to stabilize division by small numbers
    accumulation variable r = 0

    while not stopping:
        minibatch
        compute gradient g = OLS_grad
        accumulate squared gradient r = decay_rate * r + (1-decay_rate) * g @ g
        compute update update = (global_eta / small_constant + np.sqrt(r)) @ g
        apply theta = theta + update
    '''
        # Initialize weights for gradient descent
    theta_ridge = np.zeros(np.shape(X)[1]) #initial theta
    n = X.shape[0]
    decay_rate = 0
    r = 0 #gradient accumulation variable
    delta = 10e-6 #stabilize division by small numbers
    # Gradient descent loop

    for t in range(num_iters):

        # Compute gradients for OSL
        grad_Ridge = (2.0/n)*X.T @ (X @ theta_ridge - y)

        #accumulate squared gradient 
        r = decay_rate * r + (1-decay_rate) * (grad_Ridge * grad_Ridge)

        #compute update
        update = (eta / (delta + np.sqrt(r))) * grad_Ridge

        # Update parameters theta
        theta_ridge += update

    return theta_ridge

def ADAM_Ridge(X,y,eta,num_iters):
    '''
    eta = 0.001 (suggested default)
    decay1 = 0.9 
    decay2 = 0.999 exponential decay rates for moment estimates 
    small constant = 10^{8} for numberical stabilization

    initial theta = 0
    s = 0, r = 0 #1st & 2nd moment variables
    t  = 0 #time step

    while not stopping:
        minibatch
        compute gradient g = OLS_grad
        update biased first moment estimate s = decay1 * s  + (1-decay1)*g
        update biased second moment r = decay2*r + (1-decay2)*g

        correct bias in 1st mement s_debias = s / (1-decay1^{t})
        correct bias in 2nd moment r_debias = r / (1-decay2^{t})

        compute update = -eta*(s_debias/np.sqrt(r_debias)+small_constant)
        apply update theta = theta + update  
    '''
            # Initialize weights for gradient descent
    theta_Ridge = np.zeros(np.shape(X)[1]) #initial theta
    n = X.shape[0]
    decay1= 0.9
    decay2 = 0.999
    s = 0
    r = 0 #1st & 2nd moment variables 
    ts = 0 # time step
    delta = 10e-8 #numerical stabilization

    # Gradient descent loop

    for t in range(num_iters):

        # Compute gradients for OSL
        grad_Ridge = (2.0/n)*X.T @ (X @ theta_Ridge - y)

        s = decay1 * s  + (1-decay1)*grad_Ridge  
        r = decay2*r + (1-decay2)*grad_Ridge

        s_debias = s / (1-decay1**ts)
        r_debias = r / (1-decay2**ts)

        update = -eta*(s_debias/np.sqrt(r_debias)+delta)
        
        # Update parameters theta
        theta_Ridge += update

    return theta_Ridge