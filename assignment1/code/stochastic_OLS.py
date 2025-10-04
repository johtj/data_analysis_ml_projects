import numpy as np

def stochastic_GD_OLS(X,y,n,M,n_epochs,eta):
    #n = num pts
    #M = size minibatches

    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])

    for epoch in range(1,n_epochs+1): #iterations over the whole dataset


        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X[start_i:end_i] #sslice batch
            y_batch = y[start_i:end_i]
            n_batch = len(X_batch)
            #calculate gradients
            grad_OLS_batch = (2.0/n_batch)*X_batch.T @ (X_batch @ theta - y_batch)
            
            #update theta
            theta -= grad_OLS_batch*eta

    return theta


def SGD_OLS_momentum(X,y,n,M,n_epochs,eta,momentum):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])
    change  = 0.0

    for epoch in range(1,n_epochs+1): #iterations over the whole dataset

        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X[start_i:end_i] #sslice batch
            y_batch = y[start_i:end_i]
            n_batch = len(X_batch)
    
            grad = (2.0/n_batch)*X_batch.T @ (X_batch @ theta - y_batch)

            update = eta * grad + momentum*change

            # Update parameters theta
            theta -= eta*grad

            #update change 
            change = update

    return theta

    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(n_features)
    change = 0.0
    n = X.shape[0]

def SGD_OLS_ADAgrad(X,y,n,M,n_epochs,eta):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])
    r = 0 #gradient accumulation variable
    delta = 10e-7


    for epoch in range(1,n_epochs+1): #iterations over the whole dataset

    
        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X[start_i:end_i] #sslice batch
            y_batch = y[start_i:end_i]
            n_batch = len(X_batch)

            # Compute gradients for OSL
            grad = (2.0/n_batch)*X_batch.T @ (X_batch @ theta - y_batch)

            #accumulate squared gradient 
            r = r + (grad * grad)

            #compute update
            update = (eta / (delta + np.sqrt(r))) * grad

            # Update parameters theta
            theta -= update

    return theta

def SGD_OLS_RMSprop(X,y,n,M,n_epochs,eta):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])
    decay_rate = 0
    r = 0 #gradient accumulation variable
    delta = 10e-6 #stabilize division by small numbers
    


    for epoch in range(1,n_epochs+1): #iterations over the whole dataset


        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X[start_i:end_i] #sslice batch
            y_batch = y[start_i:end_i]
            n_batch = len(X_batch)
        
            grad = (2.0/n_batch)*X_batch.T @ (X_batch @ theta - y_batch)

            #accumulate squared gradient 
            r = decay_rate * r + (1-decay_rate) * (grad * grad)

            #compute update
            update = (eta / (delta + np.sqrt(r))) * grad

            # Update parameters theta
            theta += update


    return theta

def SGD_OLS_ADAM(X,y,n,M,n_epochs,eta):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])

    decay1= 0.9
    decay2 = 0.999
    s = 0
    r = 0 #1st & 2nd moment variables 
    ts = 0 # time step
    delta = 10e-8 #numerical stabilization

    for epoch in range(1,n_epochs+1): #iterations over the whole dataset


        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X[start_i:end_i] #sslice batch
            y_batch = y[start_i:end_i]
            n_batch = len(X_batch)
        
            # Compute gradients for OSL
            grad = (2.0/n_batch)*X_batch.T @ (X_batch @ theta - y_batch)
            ts += 1

            s = decay1 * s  + (1-decay1)*grad  
            r = decay2*r + (1-decay2)*grad*grad

            s_debias = s / (1-decay1**ts)
            r_debias = r / (1-decay2**ts)

            update = -eta*(s_debias/np.sqrt(r_debias)+delta)
            
            # Update parameters theta
            theta += update


    return theta

