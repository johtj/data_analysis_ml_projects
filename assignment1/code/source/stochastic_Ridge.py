import numpy as np

def stochastic_GD_Ridge(X,y,n,M,n_epochs,eta,lam):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])

    for epoch in range(1,n_epochs+1): #iterations over the whole dataset

        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X[start_i:end_i] #slice batch
            y_batch = y[start_i:end_i]

            n_batch = len(X_batch)
            #calculate gradients
            
            grad = (2.0/n_batch) * X_batch.T @(X_batch @ theta - y_batch) + 2*lam*theta            
            
            #update theta
            theta -= grad*eta

    return theta


def SGD_Ridge_momentum(X,y,n,M,n_epochs,eta,lam,momentum):
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
    
            grad = (2.0/n_batch) * X_batch.T @(X_batch @ theta - y_batch) + 2*lam*theta

            update = eta * grad + momentum*change

            # Update parameters theta
            theta -= eta*grad

            #update change 
            change = update

    return theta

    

def SGD_Ridge_ADAgrad(X,y,n,M,n_epochs,eta,lam):
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
            grad = (2.0/n_batch) * X_batch.T @(X_batch @ theta - y_batch) + 2*lam*theta

            #accumulate squared gradient 
            r = r + (grad * grad)

            #compute update
            update = (eta / (delta + np.sqrt(r))) * grad

            # Update parameters theta
            theta -= update

    return theta

def SGD_Ridge_RMSprop(X,y,n,M,n_epochs,eta,lam):
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
        
            grad = (2.0/n_batch) * X_batch.T @(X_batch @ theta - y_batch) + 2*lam*theta

            #accumulate squared gradient 
            r = decay_rate * r + (1-decay_rate) * (grad * grad)

            #compute update
            update = (eta / (delta + np.sqrt(r))) * grad

            # Update parameters theta
            theta += update


    return theta

def SGD_Ridge_ADAM(X,y,n,M,n_epochs,eta,lam):
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
            grad = (2.0/n_batch) * X_batch.T @(X_batch @ theta - y_batch) + 2*lam*theta
            ts += 1

            s = decay1 * s  + (1-decay1)*grad  
            r = decay2*r + (1-decay2)*grad*grad

            s_debias = s / (1-decay1**ts)
            r_debias = r / (1-decay2**ts)

            update = -eta*(s_debias/np.sqrt(r_debias)+delta)
            
            # Update parameters theta
            theta += update


    return theta