To run our neural network

``

from source.neural_network import NN
import source.cost_functions as cost_function
import souce.activation_functions as activation_function
import source.schedulers as scheduler
import source.utils as utils

Read and split your data into train/test.

#setup activation functions
activatition = activation_function.RELU
activation_derivative = activation_function.RELU_derivative
activation_output = activation_function.linear
activation_output_derivative = activation_function.linear_derivative

#setup cost function
cost_fn = cost_function.class_mse_loss(l1=0, l2=0) 

#pass shape for hidden layers, number of layers and number of nodes for each layer
hidden_layers = (50,100)

activations, activations_derivative, _dim = utils.create_activations_layderdim(activation, activation_derivative, 
                                                                        activation_output,activaion_output_derivative,
                                                                        HIDDEN_LAYERS, y_train, X_train)

#setup eta, number of epochs, and number of batches
ETA= 0.5  
ITERATIONS = 1000
BATCHES = 50

#setup optimizer
optimizer_ADAM = scheduler.ADAM(eta=ETA, rho=0.9, rho2=0.99) 

NN_classify = NN.NN(dims = _dim, 
                    activation_funcs = activations,
                    activation_ders = activations_derivative,
                    cost_object=cost_fn)
                    
epoch_scores, predictions = NN_classify.fit(X=X_train, 
                                            t=y_train, 
                                            epochs=ITERATIONS, 
                                            batches=BATCHES,
                                            scheduler=optimizer_ADAM,
                                            X_val=X_test,
                                            t_val=y_test)

``

