import time
import pandas as pd

import source.schedulers as schedulers
from source.cost_functions import mse

import torch
import torch.nn as nn
import torch.optim as optim

def create_activations_layderdim(activation_hidden, activation_hidden_derivative, activation_output, activation_output_derivative, hidden, target, input):
    """
    Docstring craeted with Copilot
    
    Constructs activation functions and  dimensions for a feedforward neural network.

    Parameters:
    ----------
    activation_hidden : callable
        Activation function to be used for hidden s.
    activation_hidden_derivative : callable
        Derivative of the activation function for hidden s.
    activation_output : callable
        Activation function to be used for the output .
    activation_output_derivative : callable
        Derivative of the activation function for the output .
    hidden : list of int
        List specifying the number of neurons in each hidden .
    target : np.ndarray
        Target output data. Used to determine the output  size.
    input : np.ndarray
        Input data. Used to determine the input  size.

    Returns:
    -------
    activation_functions : list of callables
        List of activation functions for each .
    activation_functions_derivative : list of callables
        List of activation function derivatives for each .
    _output_sizes : list of int
        List of output sizes for each , including input and output s.
    """

    input_dim = input.shape[1]
    output_dim = 1 if target.ndim == 1 else target.shape[1]
    if len(hidden) > 1:
        _output_sizes = [input_dim, *hidden, output_dim]
    elif len(hidden) == 1:
        _output_sizes = [input_dim, hidden[0], output_dim]
    else:
        print('Invalid length hidden layers')

    num_s = len(_output_sizes)

    #activation_functions = [activation_hidden] * (num_s - 1) + [activation_output]   # creates activation function for input layer
    #activation_functions_derivative = [activation_hidden_derivative] * (num_s - 1) + [activation_output_derivative] 

    activation_functions = [activation_hidden] * (num_s - 2) + [activation_output] 
    activation_functions_derivative = [activation_hidden_derivative] * (num_s - 2) + [activation_output_derivative] 

    return activation_functions, activation_functions_derivative, _output_sizes


def neural_network_loop(model, etas, lambdas, optimizer_name, max_iterations, x_train_scaled, y_train, x_test, y_test, cost_func = mse.cost, momentum_val=0.9, verbose=True):

    results = []

    for eta in etas:
        for lmbd in lambdas:
            
            if verbose:
                print(f"\nTraining with: optimizer={optimizer_name}, lr={eta}, lambda={lmbd}, iteration={max_iterations}")

            start_time = time.time()

            if optimizer_name == 'ADAM':
                optimizer = schedulers.ADAM(eta, rho=0, rho2=0)  
            if optimizer_name == 'ADAM_L1':
                optimizer = schedulers.ADAM(eta, rho=lmbd, rho2=0)  
            if optimizer_name == 'ADAM_L2':
                optimizer = schedulers.ADAM(eta, rho=0, rho2=lmbd)   
            elif optimizer_name == 'SGD':
                optimizer = schedulers.momentum(eta, momentum=momentum_val)
            elif optimizer_name == 'RMSprop':
                optimizer = schedulers.RMSprop(eta,rho=lmbd)

            epoch_scores, predictions = model.fit(X=x_train_scaled, 
                                                     t=y_train, 
                                                     X_val=x_test, 
                                                     t_val=y_test, 
                                                     epochs=max_iterations, 
                                                     scheduler=optimizer)
            
            final_mse = cost_func(y_true=y_test, y_pred=predictions)
            
            elapsed_time = time.time() - start_time

            results.append({
                'Learning Rate': eta,
                'Lambda': lmbd,
                'Iterations': max_iterations,
                'Elapsed Time (s)': round(elapsed_time, 2),
                'Training Errors': epoch_scores['training_errors'],
                'Validation Errors': epoch_scores['validation_errors'],
                'MSE': final_mse,
                'Predictions': predictions
            })
    return pd.DataFrame(results)


def pytorch_loop(model, cost_func, etas, lambdas, optimizer_name, max_iterations, x_train_scaled, y_train, x_test, y_test, momentum_val=0.9, verbose=True):
    """
    Copilot used to add L1 and L2 to ADAM optimizer

    Input values as tensors
    """
    results = []

    for eta in etas:
        for lmbd in lambdas:
            
            if verbose:
                print(f"\nPytorch - Training with: optimizer={optimizer_name}, lr={eta}, lambda={lmbd}, iteration={max_iterations}")

            start_time = time.time()

            if optimizer_name == 'ADAM':
                optimizer = optim.Adam(model.parameters(), lr=eta)
            if optimizer_name == 'ADAM_L1':
                optimizer = optim.Adam(model.parameters(), lr=eta) # if: see below
            if optimizer_name == 'ADAM_L2':
                optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay = lmbd) # L2 via weight_decay in Pytorch
            elif optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=eta)
            elif optimizer_name == 'RMSprop':
                optimizer = optimizer = optim.RMSprop(model.parameters(), lr=eta)

            for epoch in range(max_iterations):
                optimizer.zero_grad()
                outputs = model(x_train_scaled)
                loss = cost_func(outputs, y_train)

                if optimizer_name == 'ADAM_L1': # L1 in Pytorch
                    l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
                    loss += lmbd * l1_penalty

                loss.backward()
                optimizer.step()
            
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

            # Predict
            with torch.no_grad():
                predictions = model(x_test)
            
                mse_test = cost_func(predictions, y_test)
            
            elapsed_time = time.time() - start_time

            results.append({
                'Learning Rate': eta,
                'Lambda': lmbd,
                'Iterations': max_iterations,
                'Elapsed Time (s)': round(elapsed_time, 2),
                'MSE': mse_test.item(),
                'Predictions': predictions
            })

    return pd.DataFrame(results)
