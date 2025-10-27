import autograd.numpy as np
from source.activation_functions import activation_functions
from source.cost_functions import cost_functions


class NN:
    """
    Description: 
        Class containing the functionality for a feed forward neural network

    """
    def __init__(self,dims: tuple[int],
                 activation_func_h: callable = activation_functions.sigmoid,
                 output_func: callable = lambda x: x,
                 cost_func: callable = cost_functions.mse,
                 seed: int = None):
        
        self.dims = dims
        #need to do this so that we have one activation func per layer, not callable, but list of callables
        self.act_hidden = activation_func_h
        self.func_out = output_func
        self.cost_func = cost_func
        self.seed = seed
        
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matricies = list()
        self.classification = None

        self.reset_weights()
        self._set_classification()

    def fit():
        pass

    def predict():
        pass

    def reset_weights(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dims)-1):
            weight_array = np.random.randn(self.dims[i]+1,self.dims[i+1])
            weight_array[0,:] = np.random.randn(self.dims[i+1]) * 0.01

            self.weghts.append(weight_array)

    def _feedforward(self, X: np.ndarray):
        """
        Functionality from feed_forward_batch in original neural_network.py code

        """
    
        layers_inputs = []
        zs = []
        a = X

        for (W, b), activation_func in zip(layers, activation_funcs):
            # Normalize b to row-broadcastable shape in case it's (out_dim,1)
            if b.ndim == 2:
                if b.shape[1] == 1: 
                    b = b.ravel()         
                elif b.shape[0] == 1:
                    b = b.reshape(-1)      
                else:
                    raise ValueError(f"Bias has unexpected shape {b.shape}")
            z = a @ W + b                 
            a = activation_func(z)

            layers_inputs.append(a)
            zs.append(z)

        return layers_inputs, zs, a