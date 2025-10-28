import autograd.numpy as np
from source.activation_functions import activation_functions
from source.cost_functions import cost_functions
from typing import Callable


class NN:
    """
    Description: 
        Class containing the functionality for a feed forward neural network

    """
    def __init__(self,dims: list[int],
                 activation_func_h: list[Callable],
                 output_func: Callable = lambda x: x,
                 cost_func: Callable = cost_functions.mse,
                 seed: int = None): #if this one throws an error, switch to default val -99999 or something
        
        
        self.dims = dims #list of positive integers specifying the number of nodes
        #for each layer dims[0] gives the number of nodes for input layer dims[1] for the
        #first hidden layer and so forth, dims[-1] gives the number of nodes in the output layer

        self.act_hidden = activation_func_h #list of activation functions for hidden layers, 
        #NOTE: currently has no default, needs to add default 
        # construction as list length: len(dims)-2 of activation_functions.sigmoid

        self.func_out = output_func #callable, activation function for output layer

        self.cost_func = cost_func #callable, cost function method
        self.seed = seed #seed for np.random
        
        self.weights = list() #list of arrays where (Weights,bias) for each layer

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = None

        self.reset_weights()
       
        #elf.setup_activation_functions() #add functionality to generate default case
        #where the activation funcs are the same, not having pass only
        #self._set_classification()

    def fit(self):
        pass

    def predict(self):
        pass

    def reset_weights(self):
        """
        Initializes the weights and biases for a feed forward neural network
        of given dimensions where each layer has an array of 
        dims (layer nodes, next layer nodes)
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        """
        generates weights that are output + 1 x input where first row of 
        weights = biases
        for i in range(len(self.dims)-1):
            weight_array = np.random.randn(self.dims[i]+1,self.dims[i+1])
            weight_array[0,:] = np.random.randn(self.dims[i+1]) * 0.01

            self.weights.append(weight_array)
        """

        """ Generates list, weights, of tuples (Weight, bias) where
        weight is output n x input m
        """
        for i in range(len(self.dims)-1):
            W = np.random.randn(self.dims[i],self.dims[i+1])
            b = np.random.randn(self.dims[i+1])
            self.weights.append((W, b))




    def _feedforward(self, X: np.ndarray):
        """
        Functionality from feed_forward_batch in original neural_network.py code
        uses activation func list, that 
        """
        self.a_matrices = list()
        self.z_matrices = list()
    
        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        for (W, b), activation_func in zip(self.weights, self.act_hidden):
            # Normalize b to row-broadcastable shape in case it's (out_dim,1)
            if b.ndim == 2:
                if b.shape[1] == 1: 
                    b = b.ravel()         
                elif b.shape[0] == 1:
                    b = b.reshape(-1)      
                else:
                    raise ValueError(f"Bias has unexpected shape {b.shape}")
                
            z = a @ W + b    
            self.z_matrices.append(z)

            a = activation_func(z)
            self.a_matrices.append(a)


        return a
    
  
    def backpropagation_batch(self,inputs, predictions , targets, activation_ders, cost_der = cost_functions.mse_derivative):
        # Use the existing feed_forward_batch to get intermediate values

        # Add the original inputs to the beginning of layer_inputs
        layer_inputs = [inputs] + self.a_matrices

        layer_grads = [None] * len(self.weights)
        #dC_da = cost_der(predictions, targets)

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.weights))):
            layer_input, z, activation_der = layer_inputs[i], self.z_matrices[i], activation_ders[i]

            if i == len(self.weights) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = cost_der(predictions, targets)  
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.weights[i + 1]
                dC_da = dC_dz @ W.T 

            dC_dz = dC_da * activation_der(z)
            dC_dW = layer_input.T @ dC_dz
            #dC_dW = np.outer(dC_dz, layer_input)
            #dC_db = dC_dz
            dC_db = np.sum(dC_dz, axis=0) 

            layer_grads[i] = (dC_dW, dC_db)

            return layer_grads
            
