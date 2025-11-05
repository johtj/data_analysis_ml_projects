# Settings and preprocessing of Runge function
# Ensures that same settings and preprocessing are used in both part B and C

# imports
import autograd.numpy as np 
from sklearn.model_selection import train_test_split

# custom imports
from scaling import standard_scaler


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

