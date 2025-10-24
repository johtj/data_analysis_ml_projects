# Constants to be used in processing of data
# Ensures that same values are used


import autograd.numpy as np 


DATAPOINTS = 1000
STANDARD_DEVIATION = 0.1
TEST_SPLIT = 0.2
TRAIN_SPLIT = 1 - TEST_SPLIT
TEST_TRAIN_RANDOM_STATE = 42 # ensure reproducibility train_test_split
NP_RANDOM_SEED = 250 # ensure reproducibility numpy

ETA_VALUES = [0.1, 0.01, 0.001, 0.0001]
LAMBDA_VALUES = np.logspace(-2, -4, 10)

VERBOSE = False

# Defining Runge function 
def runge_function(x, n_datapoints=DATAPOINTS, standard_deviation=STANDARD_DEVIATION):
    y = 1 / (1 + 25 * x**2) + np.random.normal(0, standard_deviation, n_datapoints)
    return y


