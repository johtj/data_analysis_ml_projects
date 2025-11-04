# Settings and preprocessing of Runge function
# Ensures that same settings and preprocessing are used in both part B and C

# imports
import autograd.numpy as np 
from sklearn.model_selection import train_test_split

# custom imports
from scaling import standard_scaler
import activation_functions

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

VERBOSE = False

RUNGE_HIDDEN_LAYERS = (50, 100)
RUNGE_MAX_ITERATIONS = 10

ACTIVATION_FUNCTION_HIDDEN = activation_functions.sigmoid().sigmoid_func
ACTIVATION_FUNCTION_HIDDEN_DERIVATIVE = activation_functions.sigmoid().sigmoid_derivative
ACTIVATION_FUNCTION_OUTPUT = activation_functions.linear().linear
ACTIVATION_FUNCTION_OUTPUT_DERIVATIVE = activation_functions.linear().linear_derivative

    

# Defining Runge function 
def runge_function(x, n_datapoints=DATAPOINTS, standard_deviation=STANDARD_DEVIATION):
    y = 1 / (1 + 25 * x**2) + np.random.normal(0, standard_deviation, n_datapoints)
    return y



## --- Preprocessing --- 
# Generate data for Runge function
x = np.linspace(-1, 1, DATAPOINTS)
np.random.seed(NP_RANDOM_SEED)
y_noise = runge_function(x)
np.random.seed(NP_RANDOM_SEED)
y = runge_function(x, n_datapoints=DATAPOINTS, standard_deviation=0) # override standard deviation to get true function



# check to test back propagation and gradient computation in part c with more features
test_two_features = False
two_features_two_predictors = False


if test_two_features:
    np.random.seed(27) # use different seed for different numbers
    noise = np.random.normal(0, 0.13, x.shape)
    x_noisy = x + noise
    x_noise = np.vstack((x, x_noisy)).T
    x = x_noise

if two_features_two_predictors:
    y_noise2 = runge_function(x, standard_deviation=0.07) # and different standard deviation
    y_noise = np.vstack((y_noise, y_noise2)).T
    np.random.seed(27) # use different seed for different numbers
    x2 = np.linspace(-1, 1, DATAPOINTS)
    noise = np.random.normal(0, 0.13, x.shape)
    x_noisy = x + noise

    x_noise = np.vstack((x, x2)).T


    x = x_noise


# preprosessing data
x_train, x_test, y_train, y_test = train_test_split(x, y_noise, test_size=TEST_SPLIT, random_state=TEST_TRAIN_RANDOM_STATE)

# scaling of x_train and x_test
x_train_scaled, x_test_scaled, x_train_mean, x_train_std = standard_scaler(x_train, x_test) # --> verified too give same results as sklearn StandardScaler for x_train



print('Before reshaping')
print('x', x.shape)
print('y_noise', y_noise.shape)

print('x_train', x_train.shape)
print('y_train', y_train.shape)


print('x_train_scaled', x_train_scaled.shape)
print('x_test_scaled', x_test_scaled.shape)
print('y_train', y_train.shape)


# reshaping
if test_two_features:
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_noise = np.array(y_noise).reshape(-1, 1)
elif two_features_two_predictors:
    print('All good')
else:
    # Reshape for use in neural network code
    x_train_scaled = np.array(x_train_scaled).reshape(-1,1)     
    x_test_scaled = np.array(x_test_scaled).reshape(-1,1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

print('After reshaping')
print('x', x.shape)
print('y_noise', y_noise.shape)

print('x_train', x_train.shape)
print('y_train', y_train.shape)


print('x_train_scaled', x_train_scaled.shape)
print('x_test_scaled', x_test_scaled.shape)
print('y_train', y_train.shape)


def create_activations_layderdim(activation_hidden, activation_hidden_derivative, activation_output, activation_output_derivative, hidden_layers, target, input):
    """
        Docstring craeted with Copilot
        
        Constructs activation functions and layer dimensions for a feedforward neural network.

        Parameters:
        ----------
        activation_hidden : callable
            Activation function to be used for hidden layers.
        activation_hidden_derivative : callable
            Derivative of the activation function for hidden layers.
        activation_output : callable
            Activation function to be used for the output layer.
        activation_output_derivative : callable
            Derivative of the activation function for the output layer.
        hidden_layers : list of int
            List specifying the number of neurons in each hidden layer.
        target : np.ndarray
            Target output data. Used to determine the output layer size.
        input : np.ndarray
            Input data. Used to determine the input layer size.

        Returns:
        -------
        activation_functions : list of callables
            List of activation functions for each layer.
        activation_functions_derivative : list of callables
            List of activation function derivatives for each layer.
        layer_output_sizes : list of int
            List of output sizes for each layer, including input and output layers.
        """

    input_dim = input.shape[1]
    output_dim = 1 if target.ndim == 1 else target.shape[1] 
    layer_output_sizes = [input_dim, *hidden_layers, output_dim]

    num_layers = len(layer_output_sizes)
    activation_functions = [activation_hidden] * (num_layers - 1) + [activation_output] 
    activation_functions_derivative = [activation_hidden_derivative] * (num_layers - 1) + [activation_output_derivative] 

    return activation_functions, activation_functions_derivative, layer_output_sizes

activations, activations_derivative, layer_dim = create_activations_layderdim(ACTIVATION_FUNCTION_HIDDEN, ACTIVATION_FUNCTION_HIDDEN_DERIVATIVE, ACTIVATION_FUNCTION_OUTPUT, ACTIVATION_FUNCTION_OUTPUT_DERIVATIVE, RUNGE_HIDDEN_LAYERS, y_train, x_train_scaled)
