import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matrix_creation import polynomial_features, scale_features_by_intercept_use
from main_methods import OLS_parameters,Ridge_parameters
from errors import MSE,R2
from GD_OLS import gradient_descent_OLS,gradient_descent_OLS_momentum,ADAgrad_OLS,RMSprop_OLS,ADAM_OLS
from GD_Ridge import gradient_descent_ridge,gradient_descent_ridge_momentum,ADAgrad_Ridge,RMSprop_Ridge,ADAM_Ridge
from stochastic_OLS import stochastic_GD_OLS,SGD_OLS_momentum,SGD_OLS_ADAgrad,SGD_OLS_RMSprop,SGD_OLS_ADAM
from stochastic_Ridge import stochastic_GD_Ridge,SGD_Ridge_momentum,SGD_Ridge_ADAgrad,SGD_Ridge_RMSprop,SGD_Ridge_ADAM

def setup_preamble(n_datapoints,standard_deviation,p,use_intercept):
    np.random.seed(250)  # ensure reproducibility numpy
    random_state_int = 42   # ensure reproducibility train_test_split

    # generating data without noise
    x = np.linspace(-1, 1, num=n_datapoints)
    y = 1 / (1 + 25 * x**2)

    # generating data with noise
    x_noise = np.linspace(-1, 1, num=n_datapoints) + np.random.normal(0, standard_deviation, n_datapoints)
    y_noise = 1 / (1 + 25 * x_noise**2)

    x_train,x_test, y_train_origin, y_test_origin = train_test_split(x,y,test_size=0.2,random_state=random_state_int)

    X = polynomial_features(x, p,intercept=use_intercept) # intercept=True gives intercept column = 0 in standard scaler if intercept is True, and hence division by 0. 

    # test and train dataset, and scaling of X_train and X_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=random_state_int)
    X_train_scaled, X_test_scaled = scale_features_by_intercept_use(X_train, X_test, use_intercept)

    #ADD NOISY DATAAAA

    return x,y,x_train,x_test,X_train_scaled,X_test_scaled,X_train,X_test,y_train,y_test

def plot_mse(n_datapoints, x_axis, x_axis_label, mse_train, mse_test, noise=False):
    """
    Plots the Mean Squared Error (MSE) for different polynomial degrees.
    
    Returns
    -------
    Saves and shows a plot of MSE for training and test sets.

    Parameters
    ----------

    n_datapoints : int
        number of data points

    polynomial degree : int
        polynomial degree for regression

    mse_train : list
        list of MSE values for training set
    
    mse_test : list
        list of MSE values for test set

    noise : Bool
        Bool to determine if noise is included or not in dataset:
    """

    if noise:
        text = f'MSE for Different {x_axis_label} with Noise\nNumber of data points: {n_datapoints}'
        filename = f'MSE for Different {x_axis_label} with Noise - Number of data points {n_datapoints}.png'
        plt.title(text)
    else:
        text = f'MSE for Different {x_axis_label} without Noise\nNumber of data points: {n_datapoints}'
        filename = f'MSE for Different {x_axis_label} without Noise - Number of data points {n_datapoints}.png'
        plt.title(text)
    plt.plot(x_axis, mse_train, 'o-',label='MSE train')
    plt.plot(x_axis, mse_test, 'o-', label='MSE test')
    plt.xlabel(x_axis_label)
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()



def plot_r2(n_datapoints, x_axis,x_axis_label, r2_train, r2_test, noise=False):
    """
    Plots the R2 Score for different polynomial degrees.
    
    Returns
    -------
    Saves and shows a plot of R2 Score for training and test sets.

    Parameters
    ----------

    n_datapoints : int
        number of data points

    polynomial degree : int
        polynomial degree for regression

    mse_train : list
        list of MSE values for training set
    
    mse_test : list
        list of MSE values for test set

    noise : Bool
        Bool to determine if noise is included or not in dataset:
    """
        
    if noise:
        plt.title(f'R2 Score for Different {x_axis_label} with Noise\nNumber of data points: {n_datapoints}')
        filename = f'R2 for Different {x_axis_label} with Noise - Number of data points {n_datapoints}.png'
    else:
        plt.title(f'R2 Score for Different {x_axis_label} without Noise\nNumber of data points: {n_datapoints}')
        filename = f'R2 for Different {x_axis_label} without Noise - Number of data points {n_datapoints}.png'
    plt.plot(x_axis, r2_train,'o-', label='R2 train')
    plt.plot(x_axis, r2_test,'o-', label='R2 test')
    plt.xlabel(x_axis_label)
    plt.ylabel('R2 Score')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()


def explore_polynomial_degree(X_train, X_test, y_train, y_test, p, use_intercept, verbose=False):
    """
    Explores the effect of polynomial degree on MSE and R2 for
    both training and test datasets using OLS regression.

    Returns
    -------

    polynomial_degree: list
        list of polynomial degrees explored
    
    mse_train: list
        list of MSE values for training data
    
    mse_test: list
        list of MSE values for test data

    r2_train: list
        list of R2 values for training data

    r2_test: list
        list of R2 values for test data

    Parameters
    ----------
    X_train : numpy array shape (n,f)
        Training feature matrix

    X_test : numpy array shape (n,f)
        Test feature matrix   

    y_train : numpy array shape (n)
        Training target vector

    y_test : numpy array shape (n)
        Test target vector
    
    p : int
        maximum polynomial degree to explore
    
    use_intercept : Bool
        Bool to determine if intercept should be included or not in regression:
        False : no intercept 
        True : include intercept
    
    verbose : Bool
        Include verbose output from function, default set to false
   
    """

    polynomial_degree = list()
    mse_train = list()
    mse_test = list()
    r2_train = list()
    r2_test = list()

    for degree in range(1, p+1):
        polynomial_degree.append(degree)

        # Extract the relevant columns from design matrix for the current degree
        X_train_sliced = X_train[:, :degree] 
        X_test_sliced = X_test[:, :degree]
        
        # OLS Regression
        theta_OLS = OLS_parameters(X_train_sliced, y_train)
        y_tilde_train = X_train_sliced @ theta_OLS
        y_tilde_test = X_test_sliced @ theta_OLS

        # Calculate MSE for training and test data
        mse_train_OLS = MSE(y_train, y_tilde_train)
        mse_test_OLS = MSE(y_test, y_tilde_test)
        mse_train.append(mse_train_OLS)
        mse_test.append(mse_test_OLS)
        if verbose: print(f"Polynomial degree: {degree}, MSE_train_OLS: {mse_train_OLS}, MSE_test_OLS: {mse_test_OLS}")

        # Calculate R2 for training and test data
        r2_train_OLS = R2(y_train, y_tilde_train)
        r2_test_OLS = R2(y_test, y_tilde_test)
        r2_train.append(r2_train_OLS)
        r2_test.append(r2_test_OLS)
        if verbose: print(f"Polynomial degree: {degree}, R2_train_OLS: {r2_train_OLS}, R2_test_OLS: {r2_test_OLS}")


        # Sklearn Linear Regression without intercept for validation of code, test dataset only.
        # only for validation of own code        
        model = LinearRegression(fit_intercept=use_intercept)
        model.fit(X_train_sliced, y_train)
        y_pred_sklearn = model.predict(X_test_sliced)
        mse_sklearn = MSE(y_test, y_pred_sklearn)
        r2_sklearn = R2(y_test, y_pred_sklearn)

        if verbose:
            print(f"Polynomial degree: {degree}, Sklearn test R2: {r2_sklearn}, Sklearn test MSE: {mse_sklearn}")
            print(f"Polynomial degree: {degree}, R2 test: Own - sklearn {r2_test_OLS - r2_sklearn}, MSE test: Own - sklearn {mse_test_OLS - mse_sklearn}")
            print('\n') # just to add line shift between different degrees in output
        
    return polynomial_degree, mse_train, mse_test, r2_train, r2_test


def explore_lambda(X_train, X_test, y_train, y_test, lambd,n=50,verbose=False):
    """
    Explores the effect of polynomial degree on MSE and R2 for
    both training and test datasets using OLS regression.

    Returns
    -------

    lambdas: list
        list of lambda values explored
    
    mse_train: list
        list of MSE values for training data
    
    mse_test: list
        list of MSE values for test data

    r2_train: list
        list of R2 values for training data

    r2_test: list
        list of R2 values for test data

    Parameters
    ----------
    X_train : numpy array shape (n,f)
        Training feature matrix

    X_test : numpy array shape (n,f)
        Test feature matrix   

    y_train : numpy array shape (n)
        Training target vector

    y_test : numpy array shape (n)
        Test target vector
    
    lambd : tuple (int,int)
        assuming lambdas will be generated using 
        np.logspace
    
    n : int
        number of generated values for lambda, default set to 50
    """

    lambdas = []
    mse_train = []
    mse_test = []
    r2_train  = []
    r2_test = []

    
    for l in np.logspace(lambd[0],lambd[1],n):
        lambdas.append(l)

        # Apply ridge regression
        theta_ridge = Ridge_parameters(X_train, y_train,l)
        y_tilde_train = X_train @ theta_ridge
        y_tilde_test = X_test @ theta_ridge

        # Calculate MSE for training and test data
        mse_train_ridge = MSE(y_train, y_tilde_train)
        mse_test_ridge = MSE(y_test, y_tilde_test)
        mse_train.append(mse_train_ridge)
        mse_test.append(mse_test_ridge)
        if verbose: print(f"Lambda: {l}, MSE_train_ridge: {mse_train_ridge}, MSE_test_OLS: {mse_test_ridge}")

        # Calculate R2 for training and test data
        r2_train_ridge = R2(y_train, y_tilde_train)
        r2_test_ridge = R2(y_test, y_tilde_test)
        r2_train.append(r2_train_ridge)
        r2_test.append(r2_test_ridge)
        if verbose: print(f"Lambda: {l}, R2_train_ridge: {r2_train_ridge}, R2_test_ridge: {r2_test_ridge}")


    
    return lambdas, mse_train, mse_test, r2_train, r2_test


def explore_iterations_GD_methods_ridge(ns,eta,lam,x,y,x_train,x_test,X_train,X_test,y_train):

    for n in ns:

        theta = gradient_descent_ridge(X_train,y_train,eta,lam,n)
        theta_n = Ridge_parameters(X_train,y_train,lam)
        theta_m = gradient_descent_ridge_momentum(X_train,y_train,eta,lam,n,0.989)
        theta_ad = ADAgrad_Ridge(X_train,y_train,eta,lam,n)
        theta_rms = RMSprop_Ridge(X_train,y_train,eta,lam,n)
        theta_ADAM = ADAM_Ridge(X_train,y_train,eta,lam,n)


        y_tilde = X_test @ theta
        y_tilde_c = X_test @ theta_n
        y_tilde_m = X_test @ theta_m
        y_tilde_ad = X_test @ theta_ad
        y_tilde_ADAM = X_test @ theta_ADAM

        fig = plt.figure(figsize=(20,10))

        ax1 = plt.subplot(1,2,1)
        ax1.plot(x,y,label="Runges function")

        ax1.scatter(x_test,y_tilde,label="gradient descent")
        ax1.scatter(x_test,y_tilde_c,label="regular ridge")
        ax1.scatter(x_test,y_tilde_m,label="with momentum")
        ax1.scatter(x_test,y_tilde_ad,label="adagrad")
        ax1.scatter(x_test,y_tilde_ADAM,label="ADAM")

        ax1.legend()
        ax1.set_title("Test")

        y_tilde_t = X_train @ theta
        y_tilde_ct = X_train @ theta_n
        y_tilde_mt = X_train @ theta_m
        y_tilde_adt = X_train @ theta_ad
        y_tilde_ADAMt = X_train @ theta_ADAM

        ax2 = plt.subplot(1,2,2)
        ax2.plot(x,y,label="target")

        ax2.scatter(x_train,y_tilde_t,label="gradient descent")
        ax2.scatter(x_train,y_tilde_ct,label="regular ridge")
        ax2.scatter(x_train,y_tilde_mt,label="with momentum")
        ax2.scatter(x_train,y_tilde_adt,label="adagrad")
        ax2.scatter(x_train,y_tilde_ADAMt,label="ADAM")

        ax2.legend()
        ax2.set_title("Train")

        fig.suptitle(f"Results for various gradient descent using ridge regression \n eta: {eta}, number of iterations: {n}")
        fig.savefig(f"../figures/gradient_descent/gradient_descent_ridge_{n}_iterations.png")

def explore_iterations_GD_methods_OLS(ns,eta,x,y,x_train,x_test,X_train,X_test,y_train):
    for n in ns:

        theta = gradient_descent_OLS(X_train,y_train,eta,n)
        theta_n = OLS_parameters(X_train,y_train)
        theta_m = gradient_descent_OLS_momentum(X_train,y_train,eta,n,0.989)
        theta_ad = ADAgrad_OLS(X_train,y_train,eta,n)
        theta_rms = RMSprop_OLS(X_train,y_train,eta,n)
        theta_ADAM = ADAM_OLS(X_train,y_train,eta,n)


        y_tilde = X_test @ theta
        y_tilde_c = X_test @ theta_n
        y_tilde_m = X_test @ theta_m
        y_tilde_ad = X_test @ theta_ad
        y_tilde_ADAM = X_test @ theta_ADAM

        fig = plt.figure(figsize=(20,10))

        ax1 = plt.subplot(1,2,1)
        ax1.plot(x,y,label="target")

        ax1.scatter(x_test,y_tilde,label="gradient descent")
        ax1.scatter(x_test,y_tilde_c,label="regular ridge")
        ax1.scatter(x_test,y_tilde_m,label="with momentum")
        ax1.scatter(x_test,y_tilde_ad,label="adagrad")
        ax1.scatter(x_test,y_tilde_ADAM,label="ADAM")

        ax1.legend()
        ax1.set_title("Test")

        y_tilde_t = X_train @ theta
        y_tilde_ct = X_train @ theta_n
        y_tilde_mt = X_train @ theta_m
        y_tilde_adt = X_train @ theta_ad
        y_tilde_ADAMt = X_train @ theta_ADAM

        ax2 = plt.subplot(1,2,2)
        ax2.plot(x,y,label="target")

        ax2.scatter(x_train,y_tilde_t,label="gradient descent")
        ax2.scatter(x_train,y_tilde_ct,label="regular ridge")
        ax2.scatter(x_train,y_tilde_mt,label="with momentum")
        ax2.scatter(x_train,y_tilde_adt,label="adagrad")
        ax2.scatter(x_train,y_tilde_ADAMt,label="ADAM")

        ax2.legend()
        ax2.set_title("Train")

        fig.suptitle(f"Results for various gradient descent using OLS \n eta: {eta}, number of iterations: {n}")
        fig.savefig(f"../figures/gradient_descent/gradient_descent_OLS_{n}_iterations.png")


def explore_n_epochs_stochasticGD_ridge(num_epochs,num_points,size_minibatch,eta,lam,x,y,x_train,x_test,X_train,X_test,y_train):

    for epochs in num_epochs:

        theta = stochastic_GD_Ridge(X_train,y_train,num_points,size_minibatch,epochs,eta,lam)
        theta_n = Ridge_parameters(X_train,y_train,lam)
        theta_m = SGD_Ridge_momentum(X_train,y_train,num_points,size_minibatch,epochs,eta,lam,0.989)
        theta_ad = SGD_Ridge_ADAgrad(X_train,y_train,num_points,size_minibatch,epochs,eta,lam)
        theta_rms = SGD_Ridge_RMSprop(X_train,y_train,num_points,size_minibatch,epochs,eta,lam)
        theta_ADAM = SGD_Ridge_ADAM(X_train,y_train,num_points,size_minibatch,epochs,eta,lam)


        y_tilde = X_test @ theta
        y_tilde_c = X_test @ theta_n
        y_tilde_m = X_test @ theta_m
        y_tilde_ad = X_test @ theta_ad
        y_tilde_ADAM = X_test @ theta_ADAM

        fig = plt.figure(figsize=(20,10))

        ax1 = plt.subplot(1,2,1)
        ax1.plot(x,y,label="target")

        ax1.scatter(x_test,y_tilde,label="gradient descent")
        ax1.scatter(x_test,y_tilde_c,label="regular ridge")
        ax1.scatter(x_test,y_tilde_m,label="with momentum")
        ax1.scatter(x_test,y_tilde_ad,label="adagrad")
        ax1.scatter(x_test,y_tilde_ADAM,label="ADAM")

        ax1.legend()
        ax1.set_title("Test")

        y_tilde_t = X_train @ theta
        y_tilde_ct = X_train @ theta_n
        y_tilde_mt = X_train @ theta_m
        y_tilde_adt = X_train @ theta_ad
        y_tilde_ADAMt = X_train @ theta_ADAM

        ax2 = plt.subplot(1,2,2)
        ax2.plot(x,y,label="target")

        ax2.scatter(x_train,y_tilde_t,label="gradient descent")
        ax2.scatter(x_train,y_tilde_ct,label="regular ridge")
        ax2.scatter(x_train,y_tilde_mt,label="with momentum")
        ax2.scatter(x_train,y_tilde_adt,label="adagrad")
        ax2.scatter(x_train,y_tilde_ADAMt,label="ADAM")

        ax2.legend()
        ax2.set_title("Train")

        fig.suptitle(f"Results for various stochastic gradient descent using ridge regression \n eta: {eta}, number of epochs: {epochs}")
        fig.savefig(f"../figures/gradient_descent/gradient_descent_ridge_stochastic_{epochs}_epochs.png")
        
def explore_n_epochs_stochasticGD_OLS(num_epochs,num_points,size_minibatch,eta,x,y,x_train,x_test,X_train,X_test,y_train):
    
    for epochs in num_epochs:

        theta = stochastic_GD_OLS(X_train,y_train,num_points,size_minibatch,epochs,eta)
        theta_n = OLS_parameters(X_train,y_train)
        theta_m = SGD_OLS_momentum(X_train,y_train,num_points,size_minibatch,epochs,eta,0.989)
        theta_ad = SGD_OLS_ADAgrad(X_train,y_train,num_points,size_minibatch,epochs,eta)
        theta_rms = SGD_OLS_RMSprop(X_train,y_train,num_points,size_minibatch,epochs,eta)
        theta_ADAM = SGD_OLS_ADAM(X_train,y_train,num_points,size_minibatch,epochs,eta)


        y_tilde = X_test @ theta
        y_tilde_c = X_test @ theta_n
        y_tilde_m = X_test @ theta_m
        y_tilde_ad = X_test @ theta_ad
        y_tilde_ADAM = X_test @ theta_ADAM

        fig = plt.figure(figsize=(20,10))

        ax1 = plt.subplot(1,2,1)
        ax1.plot(x,y,label="target")

        ax1.scatter(x_test,y_tilde,label="gradient descent")
        ax1.scatter(x_test,y_tilde_c,label="regular ridge")
        ax1.scatter(x_test,y_tilde_m,label="with momentum")
        ax1.scatter(x_test,y_tilde_ad,label="adagrad")
        ax1.scatter(x_test,y_tilde_ADAM,label="ADAM")

        ax1.legend()
        ax1.set_title("Test")

        y_tilde_t = X_train @ theta
        y_tilde_ct = X_train @ theta_n
        y_tilde_mt = X_train @ theta_m
        y_tilde_adt = X_train @ theta_ad
        y_tilde_ADAMt = X_train @ theta_ADAM

        ax2 = plt.subplot(1,2,2)
        ax2.plot(x,y,label="target")

        ax2.scatter(x_train,y_tilde_t,label="gradient descent")
        ax2.scatter(x_train,y_tilde_ct,label="regular ridge")
        ax2.scatter(x_train,y_tilde_mt,label="with momentum")
        ax2.scatter(x_train,y_tilde_adt,label="adagrad")
        ax2.scatter(x_train,y_tilde_ADAMt,label="ADAM")

        ax2.legend()
        ax2.set_title("Train")

        fig.suptitle(f"Results for various stochastic gradient descent using OLS \n eta: {eta}, number of epochs: {epochs}")
        fig.savefig(f"../figures/gradient_descent/gradient_descent_OLS_stochastic_{epochs}_epochs.png")