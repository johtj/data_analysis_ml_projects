# CODE FOR CUSTOM MADE NEURAL NETWORK


# imports
import autograd.numpy as np 

# custom imports
from cost_functions import mse_loss_gradient #mse_derivative



# DELETE ALL NOT BATCH VERSIONS AFTER TESTING ??????



def create_layers(network_input_size, layer_output_sizes):
    """
    Initializes the weights and biases for a feedforward neural network.

    Parameters:
    ----------
    network_input_size : int
        The size of the input layer (number of input features).
    
    layer_output_sizes : list of int
        A list specifying the number of neurons in each subsequent layer of the network.

    Returns:
    -------
    layers : list of tuples
        A list where each element is a tuple (W, b) representing the weights and biases
        for a layer. W is a NumPy array of shape (layer_output_size, input_size),
        and b is a NumPy array of shape (layer_output_size,).
    """
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers



def create_layers_batch(network_input_size, layer_output_sizes):
    """
    Create layers for batched feed-forward where inputs are (batch_size, in_dim).
    Each layer stores (W, b) with:
      - W shape: (in_dim, out_dim)  [TRANSPOSED relative to single input case]
      - b shape: (out_dim,)
    """
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        # Transposed compared to single-vector case:
        # single-vector used W: (out_dim, in_dim)
        # batch version uses W: (in_dim, out_dim)
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers



def feed_forward(input, layers, activation_funcs):
    """
    Performs a forward pass through a feedforward neural network while saving intermediate values.

    Parameters:
    ----------
    input : np.ndarray
        The input vector to the network, typically of shape (input_size,).
    
    layers : list of tuples
        A list of (W, b) tuples representing the weights and biases for each layer.
        - W: Weight matrix of shape (layer_output_size, input_size)
        - b: Bias vector of shape (layer_output_size,)
    
    activation_funcs : list of callable
        A list of activation functions to apply after each layer's linear transformation.

    Returns:
    -------
    layer_inputs : list of np.ndarray
        The input activations to each layer before the linear transformation.
    
    zs : list of np.ndarray
        The linear combinations (z = W @ a + b) computed at each layer.
    
    a : np.ndarray
        The final output of the network after all layers and activations.
    """

    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = W @ a + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a


# from exercise 5 - week 41, added saving of intermediate values
def feed_forward_batch(inputs, layers, activation_funcs):
    """
    Dimensionality check with Copilot

    Perform a forward pass through a batched feed-forward neural network.

    This function assumes the input has shape (batch_size, in_dim), and each layer
    is defined by a tuple (W, b), where:
      - W has shape (in_dim, out_dim)
      - b has shape (out_dim,) or (out_dim, 1)

    Parameters
    ----------
    inputs : np.ndarray
        Input data of shape (batch_size, in_dim).
    layers : list of tuples
        Each tuple contains (W, b) for a layer:
            - W: weight matrix of shape (in_dim, out_dim)
            - b: bias vector of shape (out_dim,) or (out_dim, 1)
    activation_funcs : list of callables
        Activation functions to apply after each layer's linear transformation.

    Returns
    -------
    layers_inputs : list of np.ndarray
        The activations (outputs) from each layer after applying the activation function.
    zs : list of np.ndarray
        The linear combinations (z = a @ W + b) before activation at each layer.
    a : np.ndarray
        The final output of the network after the last activation function.
    """

    layers_inputs = []
    zs = []
    a = inputs  
    for (W, b), activation_func in zip(layers, activation_funcs):
        # Normalize b to row-broadcastable shape in case it's (out_dim,1)
        # Ensure W is 2D
        if W.ndim != 2:
            raise ValueError(f"Weight matrix W must be 2D, got shape {W.shape}")

        # Ensure b is 1D and broadcastable
        if b.ndim == 2:
            if b.shape[1] == 1:
                b = b.ravel()
            elif b.shape[0] == 1:
                b = b.reshape(-1)
            else:
                raise ValueError(f"Bias b has unexpected shape {b.shape}")
        elif b.ndim != 1:
            raise ValueError(f"Bias b must be 1D or 2D, got shape {b.shape}")

        # Check matrix multiplication compatibility
        if a.shape[1] != W.shape[0]:
            raise ValueError(f"Incompatible shapes for matrix multiplication: {a.shape} @ {W.shape}")

        z = a @ W + b                 
        a = activation_func(z)

        layers_inputs.append(a)
        zs.append(z)

    return layers_inputs, zs, a


def backpropagation(input, layers, activation_funcs, target, activation_ders, cost_der=mse_loss_gradient):
    
    """
    Computes the gradients of the cost function with respect to the weights and biases
    of a feedforward neural network using backpropagation.

    Parameters:
    ----------
    input : np.ndarray
        The input vector to the network, typically of shape (input_size,).
    
    layers : list of tuples
        A list of (W, b) tuples representing the weights and biases for each layer.
    
    activation_funcs : list of callable
        A list of activation functions applied at each layer.
    
    target : np.ndarray
        The expected output vector (ground truth) for the given input.
    
    activation_ders : list of callable
        A list of derivatives of the activation functions, corresponding to each layer.
    
    cost_der : callable, optional
        A function that computes the derivative of the cost with respect to the output
        of the network. Defaults to ``, the derivative of mean squared error.

    Returns:
    -------
    layer_grads : list of tuples
        A list of (dC_dW, dC_db) tuples representing the gradients of the cost with respect
        to the weights and biases for each layer.
        - dC_dW: Gradient of cost w.r.t. weights, same shape as W
        - dC_db: Gradient of cost w.r.t. biases, same shape as b
    """

    layer_inputs, zs, predict = feed_forward(input, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            dC_da = cost_der(predict, target)
        else:
            (W, b) = layers[i + 1]
            dC_da = W.T @ dC_dz

        dC_dz = dC_da * activation_der(z)
        dC_dW = np.outer(dC_dz, layer_input)
        dC_db = dC_dz

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads








# Batched version of backpropagation
def backpropagation_batch(inputs, layers, activation_funcs, targets, activation_ders, cost_der=mse_loss_gradient):
    # Use the existing feed_forward_batch to get intermediate values
    layer_inputs, zs, predictions = feed_forward_batch(inputs, layers, activation_funcs)

    # Add the original inputs to the beginning of layer_inputs
    layer_inputs = [inputs] + layer_inputs

    layer_grads = [None] * len(layers)

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            dC_da = cost_der(predictions, targets)  
        else:
            (W, b) = layers[i + 1]
            dC_da = dC_dz @ W.T 

        dC_dz = dC_da * activation_der(z)
        dC_dW = layer_input.T @ dC_dz
        dC_db = np.sum(dC_dz, axis=0) 

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads
