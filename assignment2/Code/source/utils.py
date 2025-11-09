import time
import pandas as pd

import source.schedulers as schedulers
from source.cost_functions import mse
import autograd.numpy as np 
import source.activation_functions as activations_functions
import source.OLS_functions as OLS_func
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from source.runge_preprocessing import NP_RANDOM_SEED


# model generation - ensure new model each iteration

def pytorch_model_mnist_fn(hidden_units_list, input_data, num_classes, pytorch_activation, image_data= False, verbose=False):
    """
    Docstring created with Copilot

    For MNIST classification

    Creates a feedforward neural network with specified hidden layers and activations.

    Parameters:
        input_data (torch.Tensor or np.ndarray): Used to determine input feature size.

    Returns:
        nn.Sequential: A PyTorch model.
    """
    input_size = input_data.shape[1]
    if image_data:
        layers = [nn.Flatten()]
    else:
        layers = []

    for hidden_unit in hidden_units_list:
        layers.append(nn.Linear(input_size, hidden_unit))
        layers.append(pytorch_activation)  
        input_size = hidden_unit

    layers.append(nn.Linear(input_size, num_classes))  # Output layer

    model = nn.Sequential(*layers)

    if verbose: print(model)  # Optional: for inspection

    return model


def pytorch_model_fn(hidden_units_list, input_data, pytorch_activation, image_data= False, verbose=False):
    """
    Docstring created with Copilot

    Creates a feedforward neural network with specified hidden layers and activations.

    Parameters:
        input_data (torch.Tensor or np.ndarray): Used to determine input feature size.

    Returns:
        nn.Sequential: A PyTorch model.
    """
    input_size = input_data.shape[1]
    if image_data:
        layers = [nn.Flatten()]
    else:
        layers = []

    for hidden_unit in hidden_units_list:
        layers.append(nn.Linear(input_size, hidden_unit))
        layers.append(pytorch_activation)  
        input_size = hidden_unit

    layers.append(nn.Linear(input_size, 1))  # Output layer

    model = nn.Sequential(*layers)

    if verbose: print(model)  # Optional: for inspection

    return model




def rescale_predictions(results, y_train_std, y_train_mean):
    min_index = results['MSE test'].idxmin()
    best_predictions = results.loc[min_index]['Predictions']
    predictions_rescaled = OLS_func.rescale_y(best_predictions, y_train_std, y_train_mean)
    return predictions_rescaled


def decide_activation_func(ACTIVATION_FUNC):
    ACTIVATION_FUNCTION_OUTPUT = activations_functions.linear
    ACTIVATION_FUNCTION_OUTPUT_DERIVATIVE = activations_functions.linear_derivative

    if ACTIVATION_FUNC=='sigmoid':
        ACTIVATION_FUNCTION = activations_functions.sigmoid
        ACTIVATION_FUNCTION_DERIVATIVE = activations_functions.sigmoid_derivative
    elif ACTIVATION_FUNC=='RELU':
        ACTIVATION_FUNCTION = activations_functions.RELU
        ACTIVATION_FUNCTION_DERIVATIVE = activations_functions.RELU_derivative
    elif ACTIVATION_FUNC=='LRELU':
        ACTIVATION_FUNCTION = activations_functions.LRELU
        ACTIVATION_FUNCTION_DERIVATIVE = activations_functions.LRELU_derivative

    return ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_DERIVATIVE, ACTIVATION_FUNCTION_OUTPUT, ACTIVATION_FUNCTION_OUTPUT_DERIVATIVE




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
        List specifying the number of neurons in each hidden layer .
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

    activation_functions = [activation_hidden] * (num_s - 2) + [activation_output] # no activation for input layer
    activation_functions_derivative = [activation_hidden_derivative] * (num_s - 2) + [activation_output_derivative] 

    return activation_functions, activation_functions_derivative, _output_sizes



def neural_network_loop(model_fn, etas, lambdas, optimizer_name, max_iterations,
                        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, batch_val=None,
                        rho=None, rho2=None, momentum=None, l1_l2 = None, verbose=False):

    def _check_missing_params(): # for own implementation lamda is passed into cost function
        missing = []
        if optimizer_name == 'ADAM':
            if rho is None: missing.append("rho (beta1)")
            if rho2 is None: missing.append("rho2 (beta2)")
            if batch_val is None: missing.append("batch")
        elif optimizer_name == 'RMSprop':
            if rho is None: missing.append("rho1")
            if batch_val is None: missing.append("batch")
        elif optimizer_name == 'SGD':
            if momentum is None: missing.append("momentum")
            if batch_val is None: missing.append("batch")
        return missing

    missing = _check_missing_params()
    if missing:
        raise ValueError(f"Missing hyperparameters for {optimizer_name}: {', '.join(missing)}")

    results = []

    for eta in etas:
        for lmbd in lambdas:
            if verbose:
                print(f"\nTraining with optimizer={optimizer_name}, lr={eta}, lambda={lmbd}, iterations={max_iterations}")

            model = model_fn()

            if l1_l2 == 'L1':
                model.cost_object.l1 = lmbd
                print(f'L1 term: {model.cost_object.l1}')
            elif l1_l2 == 'L2':
                model.cost_object.l2 = lmbd
                print(f'L2 term: {model.cost_object.l2}')


            if optimizer_name == 'ADAM':
                optimizer = schedulers.ADAM(eta, rho, rho2)
            elif optimizer_name == 'SGD':
                optimizer = schedulers.momentum(eta, momentum)
            elif optimizer_name == 'RMSprop':
                optimizer = schedulers.RMSprop(eta, rho)

            start_time = time.time()

            scores, predictions = model.fit(
                X=X_train_scaled,
                t=y_train_scaled,
                X_val=X_test_scaled,
                t_val=y_test_scaled,
                epochs=max_iterations,
                scheduler=optimizer,
                batches=batch_val
            )

            elapsed_time = round(time.time() - start_time, 2)

            results.append({
                'Learning Rate': eta,
                'Lambda': lmbd,
                'Iterations': max_iterations,
                'Elapsed Time (s)': elapsed_time,
                'Training Errors': scores['training_errors'],
                'Validation Errors': scores['validation_errors'],
                'MSE train': scores['training_errors'][-1],
                'MSE test': scores['validation_errors'][-1],
                'Predictions': predictions
            })

    return pd.DataFrame(results)



def pytorch_loop(model_fn, etas, lambdas, optimizer_name, max_iterations,
                 X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, cost_func=None,
                 rho=None, rho2=None, weight_decay=None, momentum=None,
                 batch_size=None, verbose=False):

    def _check_missing_params():
        missing = []
        if optimizer_name == 'ADAM':
            if rho is None: missing.append("rho (beta1)")
            if rho2 is None: missing.append("rho2 (beta2)")
            if weight_decay is None: missing.append("weight_decay/lambda")
            if cost_func is None: missing.append("cost function")
        elif optimizer_name == 'RMSprop':
            if rho is None: missing.append("rho (decay)")
            if momentum is None: missing.append("momentum")
            if weight_decay is None: missing.append("weight_decay/lambda")
            if cost_func is None: missing.append("cost function")
        elif optimizer_name == 'SGD':
            if momentum is None: missing.append("momentum")
            if batch_size is None: missing.append("batch_size")
            if weight_decay is None: missing.append("weight_decay/lambda")
            if cost_func is None: missing.append("cost function")
        return missing

    missing = _check_missing_params()
    if missing:
        raise ValueError(f"Missing hyperparameters for {optimizer_name}: {', '.join(missing)}")

    results = []
    train_ds = TensorDataset(X_train_scaled, y_train_scaled)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if cost_func == 'MSE':
        loss_func = nn.MSELoss()
    elif cost_func == 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss()
    
    print(loss_func)

    for eta in etas:
        for lmbd in lambdas:
            if verbose:
                print(f"\nTraining with optimizer={optimizer_name}, lr={eta}, lambda={lmbd}, iterations={max_iterations}")
            
            torch.manual_seed(NP_RANDOM_SEED)
            model = model_fn()

            if optimizer_name == 'ADAM':
                optimizer = torch.optim.Adam(model.parameters(), lr=eta, betas=(rho, rho2), weight_decay=lmbd)  # weight_decay more or less lambda
            elif optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum, weight_decay=lmbd)
            elif optimizer_name == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=eta, alpha=rho, momentum=momentum, weight_decay=lmbd)


            start_time = time.time()

            for epoch in range(max_iterations):
                model.train()
                total_loss = 0.0
                for x_batch, y_batch in train_dl:
                    y_batch = y_batch.squeeze()
                    
                    optimizer.zero_grad()
                    pred = model(x_batch).squeeze()
                    loss = loss_func(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if verbose and epoch % 20 == 0:
                    avg_loss = total_loss / len(train_dl)
                    print(f"Epoch {epoch}: Training Loss = {avg_loss:.4f}")

            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_scaled)

            elapsed_time = time.time() - start_time

            if cost_func == 'MSE':
                test_loss = loss_func(test_pred.squeeze(), y_test_scaled.squeeze()).item()
                result = {
                    'Learning Rate': eta,
                    'Lambda': lmbd,
                    'Iterations': max_iterations,
                    'Elapsed Time (s)': elapsed_time,
                    'MSE test': test_loss,
                    'Predictions': test_pred.cpu().numpy()
                }

            elif cost_func == 'CrossEntropy':
                # Get predicted class labels
                predicted_classes = torch.argmax(test_pred, dim=1)
                true_classes = y_test_scaled.view(-1).long()  # Ensure correct shape and type

                # Compute accuracy
                correct = (predicted_classes == true_classes).sum().item()
                total = true_classes.size(0)
                accuracy = correct / total

                result = {
                    'Learning Rate': eta,
                    'Lambda': lmbd,
                    'Iterations': max_iterations,
                    'Elapsed Time (s)': elapsed_time,
                    'Accuracy': accuracy,
                    'Predictions': predicted_classes.cpu().numpy()
                }

            results.append(result)

    return pd.DataFrame(results)