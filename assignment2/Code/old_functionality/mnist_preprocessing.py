# Settings and preprocessing of MNIST dataset
# Ensures that same settings and preprocessing are used in part f

import autograd.numpy as np

# constants
MNIST_RANDOM_STATE = 42
TEST_SPLIT = 0.2
#TRAIN_SPLIT = 1 - TEST_SPLIT
ITERATIONS = 1000
TORCH_SEED = 1
HIDDEN_LAYERS = [32, 16]
BATCH_SIZE = 64
ETA_VALUES = [0.1, 0.01, 0.001, 0.0001]
LAMBDA_VALUES = np.logspace(-2, -4, 10)
MOMENTUM = 0.9

# Download MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

# Extract data (features) and target (labels)
X = mnist.data
y = mnist.target

# Scaling pixel values
X = X / 255.0


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=MNIST_RANDOM_STATE)


