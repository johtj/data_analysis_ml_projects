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
DATAPOINTS = 1000 # Datapoints used from proejct 1
STANDARD_DEVIATION = 0.1

TEST_SPLIT = 0.2
TRAIN_SPLIT = 1 - TEST_SPLIT

TEST_TRAIN_RANDOM_STATE = 42 # ensure reproducibility train_test_split
NP_RANDOM_SEED = 250 # ensure reproducibility numpy

ETA_VALUES = [0.1, 0.01, 0.001, 0.0001]
LAMBDA_VALUES = np.logspace(-2, -4, 10)

  
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



# split
x_train, x_test, y_train, y_test = train_test_split(x, y_noise, test_size=TEST_SPLIT, random_state=TEST_TRAIN_RANDOM_STATE)

# scaling of x_train and x_test
x_train_scaled, x_test_scaled, x_train_mean, x_train_std = standard_scaler(x_train, x_test) # --> verified too give same results as sklearn StandardScaler for x_train
y_train_scaled, y_test_scaled, y_train_mean, y_train_std = standard_scaler(y_train, y_test) 

# Reshape for use in neural network code (batch)
x_train = np.array(x_train).reshape(-1,1)     
x_train_scaled = np.array(x_train_scaled).reshape(-1,1)     
x_test_scaled = np.array(x_test_scaled).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)

y_train = np.array(y_train).reshape(-1, 1)
y_train_scaled = np.array(y_train_scaled).reshape(-1, 1)
y_test_scaled = np.array(y_test_scaled).reshape(-1, 1)



## Settings and data generation OLS

VERBOSE_POLY_DEGREE = False

USE_INTERCEPT = True
polynomial_degree = 15

import source.OLS_functions as OLS_func

## Code from project 1 to calculate OLS regression
X = OLS_func.polynomial_features(x, polynomial_degree,USE_INTERCEPT)
X_train_ols, X_test_ols, y_train_ols, y_test_ols = train_test_split(X, y_noise, test_size=TEST_SPLIT, random_state=TEST_TRAIN_RANDOM_STATE) 
# Scaleing
X_train_scaled_ols, X_test_scaled_ols, X_train_mean_ols, X_train_std_ols = OLS_func.scale_features_by_intercept_use(X_train_ols, X_test_ols, USE_INTERCEPT)
y_train_scaled_ols, y_test_scaled_ols, y_train_mean_ols, y_train_std_ols = OLS_func.standard_scaler(y_train_ols, y_test_ols)

# Train
theata_scaled = OLS_func.OLS_parameters(X_train_scaled_ols, y_train_scaled_ols)

# Predict
rescaled_theta, rescaled_intercept = OLS_func.rescale_theta_intercept(theata_scaled[1:], theata_scaled[0], y_train_std_ols, y_train_mean_ols, X_train_std_ols, X_train_mean_ols, verbose=VERBOSE_POLY_DEGREE)
y_predicted_scaled_ols = OLS_func.predict_y(X_test_scaled_ols[:, 1:], rescaled_theta)
y_predicted_rescaled_ols = OLS_func.rescale_y(y_predicted_scaled_ols, y_train_std_ols, y_train_mean_ols)
polynomial_degree, mse_train_ols, mse_test_ols, r2_train_ols, r2_test_ols, thetas_ols = OLS_func.explore_polynomial_degree(X_train_scaled_ols, X_test_scaled_ols, y_train_scaled_ols, y_test_scaled_ols, polynomial_degree, USE_INTERCEPT, verbose=VERBOSE_POLY_DEGREE)

y_train_scaled_ols = np.array(y_train_scaled_ols).reshape(-1, 1)
y_test_scaled_ols = np.array(y_test_scaled_ols).reshape(-1, 1)
