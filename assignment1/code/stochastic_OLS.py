import numpy as np

def stochastic_GD_OLS(X,y,n,n_features,M,n_epochs,eta):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])

    for epoch in range(1,n_epochs+1): #iterations over the whole dataset

        X_shuffled = np.random.permutation(X) #reshuffle x 
        y_shuffled = np.random.permutation(y) #resuffle y

        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X_shuffled[start_i:end_i] #sslice batch
            y_batch = y_shuffled[start_i:end_i]
            n_batch = len(X_batch)
            #calculate gradients
            grad_OLS_batch = (2.0/n_batch)*X_batch.T @ (X_batch @ theta - y_batch)
            
            #update theta
            theta -= grad_OLS_batch*eta

    return theta


def SGD_OLS_momentum(X,y,n,n_features,M,n_epochs,eta,momentum):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])
    change  = 0.0

    for epoch in range(1,n_epochs+1): #iterations over the whole dataset

        X_shuffled = np.random.permutation(X) #reshuffle x 
        y_shuffled = np.random.permutation(y) #resuffle y

        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X_shuffled[start_i:end_i] #sslice batch
            y_batch = y_shuffled[start_i:end_i]
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

def stochastic_GD_OLS(X,y,n,n_features,M,n_epochs,eta):
    m = int(n/M) #number of minibatches

    theta = np.zeros(np.shape(X)[1])

    for epoch in range(1,n_epochs+1): #iterations over the whole dataset

        X_shuffled = np.random.permutation(X) #reshuffle x 
        y_shuffled = np.random.permutation(y) #resuffle y

        for i in range(m): #do number of minibatches
            
            k = np.random.randint(m) #choose random minibatch
            start_i = k*M #random minibatch * minibatch size
            end_i = start_i + M -1

            X_batch = X_shuffled[start_i:end_i] #sslice batch
            y_batch = y_shuffled[start_i:end_i]
            n_batch = len(X_batch)

            

    return theta
