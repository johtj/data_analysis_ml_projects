# Settings and preprocessing of Runge function
# Ensures that same settings and preprocessing are used in both part B and C

# imports
import autograd.numpy as np 
from sklearn.model_selection import train_test_split

# custom imports
from source.scaling import standard_scaler
import source.activation_functions as activations_functions

## --- Settings --- 
# Constants
DATAPOINTS = 1000
STANDARD_DEVIATION = 0.1

TEST_SPLIT = 0.2
TRAIN_SPLIT = 1 - TEST_SPLIT

TEST_TRAIN_RANDOM_STATE = 42 # ensure reproducibility train_test_split
NP_RANDOM_SEED = 250 # ensure reproducibility numpy

ETA_VALUES = [0.1, 0.01, 0.001, 0.0001]
LAMBDA_VALUES = np.logspace(-2, -4, 10)

MOMENTUM = 0.9

VERBOSE = False

RUNGE_HIDDEN_S = (50, 100)
RUNGE_MAX_ITERATIONS = 10

ACTIVATION_FUNCTION_HIDDEN = activations_functions.sigmoid
ACTIVATION_FUNCTION_HIDDEN_DERIVATIVE = activations_functions.sigmoid_derivative
ACTIVATION_FUNCTION_OUTPUT = activations_functions.linear
ACTIVATION_FUNCTION_OUTPUT_DERIVATIVE = activations_functions.linear_derivative

    

# Defining Runge function 
def runge_function(x, n_datapoints=DATAPOINTS, standard_deviation=STANDARD_DEVIATION):
    y = 1 / (1 + 25 * x**2) + np.random.normal(0, standard_deviation, n_datapoints)
    return y


def create_activations_layderdim(activation_hidden, activation_hidden_derivative, activation_output, activation_output_derivative, hidden_s, target, input):
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
        hidden_s : list of int
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
    _output_sizes = [input_dim, *hidden_s, output_dim]

    num_s = len(_output_sizes)
    activation_functions = [activation_hidden] * (num_s - 1) + [activation_output] 
    activation_functions_derivative = [activation_hidden_derivative] * (num_s - 1) + [activation_output_derivative] 

    return activation_functions, activation_functions_derivative, _output_sizes

## --- Preprocessing --- 
# Generate data for Runge function
x = np.linspace(-1, 1, DATAPOINTS)
np.random.seed(NP_RANDOM_SEED)
y_noise = runge_function(x)
np.random.seed(NP_RANDOM_SEED)
y = runge_function(x, n_datapoints=DATAPOINTS, standard_deviation=0) # override standard deviation to get true function


# split
x_train, x_test, y_train, y_test = train_test_split(x, y_noise, test_size=TEST_SPLIT, random_state=TEST_TRAIN_RANDOM_STATE)

# scaling of x_train and x_test
x_train_scaled, x_test_scaled, x_train_mean, x_train_std = standard_scaler(x_train, x_test) # --> verified too give same results as sklearn StandardScaler for x_train

# Reshape for use in neural network code (batch)
x_train_scaled = np.array(x_train_scaled).reshape(-1,1)     
x_test_scaled = np.array(x_test_scaled).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)


activations, activations_derivative, _dim = create_activations_layderdim(ACTIVATION_FUNCTION_HIDDEN, ACTIVATION_FUNCTION_HIDDEN_DERIVATIVE, ACTIVATION_FUNCTION_OUTPUT, ACTIVATION_FUNCTION_OUTPUT_DERIVATIVE, RUNGE_HIDDEN_S, y_train, x_train_scaled)
