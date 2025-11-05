import autograd.numpy as np
from activation_functions import activation_function
from cost_functions import cost_function
from cost_functions import mse
from cost_functions import cross_entropy
from typing import Callable
import schedulers as scheduler_methods
from sklearn.utils import resample
from copy import copy


class NN:
    """
    Description: 
        Class containing the functionality for a feed forward neural network

    """
    def __init__(self,dims: list[int],
                 activation_funcs: list[Callable],
                 activation_ders: list[Callable],
                 cost: cost_function = mse(),
                 seed: int = None): #if this one throws an error, switch to default val -99999 or something
        
        
        self.dims = dims #list of positive integers specifying the number of nodes
        #for each layer dims[0] gives the number of nodes for input layer dims[1] for the
        #first hidden layer and so forth, dims[-1] gives the number of nodes in the output layer

        self.activation_funcs = activation_funcs #list of activation functions for hidden layers, 
        self.activation_ders = activation_ders
        #NOTE: currently has no default, needs to add default 
        # construction as list length: len(dims)-1 of activation_functions.sigmoid

        self.cost_func = cost.cost #callable, cost function method
        self.cost_der = cost.cost_derivative #callable, derivative of the cost function
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

    def fit(self,
            X: np.ndarray,
            t: np.ndarray,
            scheduler: scheduler_methods.scheduler,
            batches: int = 1,
            epochs: int = 100,
            lam: float = 0,
            X_val: np.ndarray = None,
            t_val: np.ndarray = None
            ):
        if self.seed is not None:
            np.random.seed(self.seed)

        val_set = False
        if X_val is not None and t_val is not None:
            val_set = True

        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        batch_size = X.shape[0] // batches

        X, t = resample(X,t) 

        cost_function_train = self.cost_func#(t)
        if val_set:
            cost_function_val = self.cost_func#(t_val)

        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))

        print(f"Using scheduler: {scheduler.__class__.__name__} with Eta={scheduler.eta}")  #LIST of schedulers, added [0] just to run code

        try: 
            for e in range(epochs):
                for i in range(batches):
                    if i == batches -1:
                        #if we are on the last batch, take everything that is left
                        X_batch = X[i*batch_size : ,:]
                        t_batch = t[i*batch_size : ,:]
                    else:
                        #regular case, use same batch size
                        X_batch = X[i*batch_size : (i+1) * batch_size, :]
                        t_batch = t[i*batch_size : (i+1) * batch_size, :]
                    self._feedforward(X_batch)
                    self._backpropagation(X_batch,t_batch)
                
                for scheduler in self.schedulers_weight:
                    scheduler.reset()

                for scheduler in self.schedulers_bias:
                    scheduler.reset()
                
                pred_train = self.predict(X)
                training_error = cost_function_train(t, pred_train) #training_error = cost_function_train(pred_train) - missing target values
                

                train_errors[e] = training_error

                #validation
                if val_set:
                    pred_val = self.predict(X_val)
                    validation_error = cost_function_val(t_val, pred_val) #validation_error = cost_function_val(pred_val) - missing target values
                    val_errors[e] = validation_error
                
                if self.classification:
                    train_accuracy = self._accuracy(self.predict(X),t)
                    train_accs[e] = train_accuracy
                    if val_set:
                        validation_accuracy = self._accuracy(pred_val,t_val)
                        val_accs[e] = validation_accuracy

        except KeyboardInterrupt:
            pass #allows for stopping at any point to see progress
        
        scores = dict()

        scores["training_errors"] = train_errors

        if val_set: 
            scores["validation_errors"] = val_errors
        
        if self.classification:
            scores["training_accuracies"] = train_accs

            if val_set:
                scores["validation_accuracies"] = val_accs

        if val_set: return scores, pred_val
        else: return scores

    def predict(self,X: np.ndarray, *, threshold=0.5):

        predict = self._feedforward(X)

        if self.classification:
            return np.where(predict > threshold, 1, 0)
        else:
            return predict

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
        output_size = self.dims[-1]
        i_size = self.dims[0]  # number of features input data
        #print(self.dims[1:])
        for layer_output_size in self.dims[1:]: # step through sizes of hidden and output layer
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            self.weights.append((W, b))  
            i_size = layer_output_size   # ensure next step in hidden and output layer



    def _feedforward(self, X: np.ndarray):
        """
        Functionality from feed_forward_batch in original neural_network.py code
        uses activation func list, that 
        """
        self.a_matrices = list()
        self.z_matrices = list()
    
        a = X

        #self.a_matrices.append(a)
        #self.z_matrices.append(a)

        for (W, b), activation_func in zip(self.weights, self.activation_funcs):
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
            self.z_matrices.append(z)

            #a = activation_func(self, X=z)
            a = activation_func(z)
            self.a_matrices.append(a)


        return a
    
  
    def _backpropagation(self, inputs, targets):
        # Use the existing feed_forward_batch to get intermediate values

        # Add the original inputs to the beginning of layer_inputs
        layer_inputs = [inputs] + self.a_matrices

        layer_grads = [None] * len(self.weights)

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.weights))):
            layer_input, z, activation_der = layer_inputs[i], self.z_matrices[i], self.activation_ders[i]

            if i == len(self.weights) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(self.a_matrices[-1], targets)  
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.weights[i + 1]
                dC_da = dC_dz @ W.T 

            #dC_dz = dC_da * self.activation_ders[i](self, X=z)
            dC_dz = dC_da * activation_der(z)

            #calculate gradients
            gradient_weights = layer_input.T @ dC_dz
            gradient_bias = np.sum(dC_dz, axis=0) 

            layer_grads[i] = (gradient_weights, gradient_bias)

            weight_array = self.weights[i][0]
            bias_array = self.weights[i][1]

            updates_weights = self.schedulers_weight[i].calculate_update(gradient_weights)
            updates_biases = self.schedulers_bias[i].calculate_update(gradient_bias)

            weight_array -= updates_weights
            bias_array -= updates_biases

            self.weights[i] = (weight_array,bias_array)

        return layer_grads
   
 
    def _accuracy(self, prediction: np.ndarray, target: np.ndarray):
        """
        Description:
        ------------
            Calculates accuracy of given prediction to target

        Parameters:
        ------------
            I   prediction (np.ndarray): vector of predicitons output network
                (1s and 0s in case of classification, and real numbers in case of regression)
            II  target (np.ndarray): vector of true values (What the network ideally should predict)

        Returns:
        ------------
            A floating point number representing the percentage of correctly classified instances.
        """
        assert prediction.size == target.size
        return np.average((target == prediction))


    def _backpropagation_for_gradient_check(self, inputs, targets):
        """
        Removed update of gradients from _backpropagation. 
        Gradient comparison is done on first backpropagation step.
        """
        # Use the existing feed_forward_batch to get intermediate values

        # Add the original inputs to the beginning of layer_inputs
        layer_inputs = [inputs] + self.a_matrices

        layer_grads = [None] * len(self.weights)

        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.weights))):
            layer_input, z, activation_der = layer_inputs[i], self.z_matrices[i], self.activation_ders[i]

            if i == len(self.weights) - 1:
                # For last layer we use cost derivative as dC_da(L) can be computed directly
                dC_da = self.cost_der(self.a_matrices[-1], targets)  
            else:
                # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.weights[i + 1]
                dC_da = dC_dz @ W.T 

            #dC_dz = dC_da * self.activation_ders[i](self, X=z)
            dC_dz = dC_da * activation_der(z)

            #calculate gradients
            gradient_weights = layer_input.T @ dC_dz
            gradient_bias = np.sum(dC_dz, axis=0) 

            layer_grads[i] = (gradient_weights, gradient_bias)

        return layer_grads
   

    def autograd_gradients(self, X, targets):
        """
        Docstring created with Copilot

        Computes gradients of the mean squared error (MSE) loss with respect to the network's weights and biases
        using Autograd's automatic differentiation.

        Parameters
        ----------
        X : ndarray
            Input data of shape (n_samples, n_features).
        targets : ndarray
            Target output values of shape (n_samples, n_outputs).

        Returns
        -------
        list of tuples
            A list containing tuples of gradients (dW, db) for each layer, where:
            - dW is the gradient of the loss with respect to the weight matrix W
            - db is the gradient of the loss with respect to the bias vector b

        Notes
        -----
        This method assumes that `self.weights` is a list of (W, b) tuples where W and b are 
        autograd-compatible arrays. It also assumes that `self.activation_funcs` is a list of 
        activation functions that accept keyword argument `X` for input.
        """

        from autograd import grad

        def forward(weights_bias, X_):
            a = X_
            for (W, b), act in zip(weights_bias, self.activation_funcs):
                z = a @ W + b 
                #a = act(self, X=z)            
                a = act(z)    
            return a

        def mse_loss(weights_bias, X_, y_):
            y_pred = forward(weights_bias, X_)
            return np.mean((y_pred - y_) ** 2) 

        weights_bias = tuple((W, b) for (W, b) in self.weights)

        loss_grad = grad(mse_loss)
        grads = loss_grad(weights_bias, X, targets)  

        return [(gW, gb) for (gW, gb) in grads]
    
    def compare_gradients(self, X, targets, atol=1e-6):  
        """
        Compares gradients computed by the model's manual backpropagation method with those
        computed using Autograd's automatic differentiation.

        Parameters
        ----------
        X : ndarray
            Input data of shape (n_samples, n_features).
        targets : ndarray
            Target output values of shape (n_samples, n_outputs).
        atol : float, optional
            Absolute tolerance used in `np.allclose` to determine if gradients match. Default is 1e-6.

        Returns
        -------
        None
            Prints a comparison of whether the gradients match for each layer, and the actual
            differences between the manually computed and Autograd-computed gradients.

        Notes
        -----
        This method is useful for debugging and validating the correctness of the manual 
        backpropagation implementation. It assumes that `self._feedforward` prepares the 
        necessary intermediate values for backpropagation, and that `self.autograd_gradients` 
        returns gradients in the same format as `self._backpropagation`.
        """
        
        self._feedforward(X)
        own_gradients = self._backpropagation_for_gradient_check(X, targets)
        
        autograd_gradients = self.autograd_gradients(X, targets)

        print()
        print('Compare gradients between own and autograd calculations:')
        for i, ((dW_manual, db_manual), (dW_auto, db_auto)) in enumerate(zip(own_gradients, autograd_gradients)):
            w_close = np.allclose(dW_manual, dW_auto, atol=atol)
            b_close = np.allclose(db_manual, db_auto, atol=atol)
            print(f"Layer {i}: dW match: {w_close}, db match: {b_close}")

        print()
        print('Differences between own and autograd gradients')
        for i, ((dW_manual, db_manual), (dW_auto, db_auto)) in enumerate(zip(own_gradients, autograd_gradients)):
            print(f"Layer {i} - dW diff:\n{dW_manual - dW_auto}")
            print(f"Layer {i} - db diff:\n{db_manual - db_auto}")
