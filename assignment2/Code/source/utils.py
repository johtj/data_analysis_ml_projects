import time
import pandas as pd

import source.schedulers as schedulers
from source.cost_functions import mse

import torch
import torch.nn as nn
import torch.optim as optim


# cost function
mse_formula = mse().cost

def neural_network_loop(model, cost_func, etas, lambdas, optimizer_name, max_iterations, x_train_scaled, y_train, x_test, y_test, momentum_val=0.9, verbose=True):#, seed=NP_RANDOM_SEED):

    results = []

    for eta in etas:
        for lmbd in lambdas:
            
            if verbose:
                print(f"\nTraining with: optimizer={optimizer_name}, lr={eta}, lambda={lmbd}, iteration={max_iterations}")

            start_time = time.time()

            if optimizer_name == 'ADAM':
                optimizer = schedulers.ADAM(eta, rho=lmbd, rho2=0)  
            if optimizer_name == 'ADAM_L1':
                optimizer = schedulers.ADAM(eta, rho=lmbd, rho2=0)  
            if optimizer_name == 'ADAM_L2':
                optimizer = schedulers.ADAM(eta, rho=0, rho2=lmbd)   
            elif optimizer_name == 'SGD':
                optimizer = schedulers.momentum(eta, momentum=momentum_val)
            elif optimizer_name == 'RMSprop':
                optimizer = schedulers.RMSprop(eta=etas,rho=lmbd)

            epoch_scores, predictions = model.fit(X=x_train_scaled, 
                                                     t=y_train, 
                                                     X_val=x_test, 
                                                     t_val=y_test, 
                                                     epochs=max_iterations, 
                                                     scheduler=optimizer)
            
            final_mse = mse_formula(y_true=y_test, y_pred=predictions)
            
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
                optimizer = optim.Adam(model.parameters(), lr=eta) # 
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
