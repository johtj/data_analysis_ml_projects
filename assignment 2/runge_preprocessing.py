# Preprocessing of Runge function data
# Ensures that same values are used in both part B and C

# imports
import autograd.numpy as np 
from sklearn.model_selection import train_test_split

# custom imports
from scaling import standard_scaler
from settings import runge_function, TEST_SPLIT, TEST_TRAIN_RANDOM_STATE, NP_RANDOM_SEED, DATAPOINTS



#####################
##################### SHOULD SETTINGS AND RUNGE PREPROCESSING BE MERGED? #####################
#####################

# Generate data for Runge function
np.random.seed(NP_RANDOM_SEED)
x = np.linspace(-1, 1, DATAPOINTS)
np.random.seed(NP_RANDOM_SEED)
y_noise = runge_function(x)
np.random.seed(NP_RANDOM_SEED)
y = runge_function(x, n_datapoints=DATAPOINTS, standard_deviation=0) # override standard deviation to get true function

# preprosessing data
x_train, x_test, y_train, y_test = train_test_split(x, y_noise, test_size=TEST_SPLIT, random_state=TEST_TRAIN_RANDOM_STATE)

# scaling of x_train and x_test
x_train_scaled, x_test_scaled, x_train_mean, x_train_std = standard_scaler(x_train, x_test) # --> verified too give same results as sklearn StandardScaler for x_train


RUNGE_HIDDEN_LAYERS = (50, 100)
RUNGE_MAX_ITERATIONS = 10