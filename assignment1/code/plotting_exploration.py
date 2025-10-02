import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from main_methods import OLS_parameters,Ridge_parameters, lasso_gradient_descent
from errors import MSE,R2

def plot_mse(regression_method, degree,  n_datapoints, x_axis_data, mse_train, mse_test, noise=False):
    """
    Plots the Mean Squared Error (MSE) for different polynomial degrees.
    
    Returns
    -------
    Saves and shows a plot of MSE for training and test sets.

    Parameters
    ----------
    regression_method : string
        Type of regression for plotting

    degree : int
        number of polynomials in regression
    
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
        noise_text = "with"
    else:
        noise_text = "without"
    
    if regression_method == "Ridge":
        x_axis_label = f"Lambdas"
        n_lambdas = len(x_axis_data)
        #text = f'{regression_method} - MSE for Different {x_axis_label} {noise_text} Noise\nNumber of lambdas {n_lambdas}\nNumber of data points: {n_datapoints}\nNumber of polynomials {degree}'
        filename = f'{regression_method} - MSE for Different {x_axis_label} {noise_text} Noise - Number of lambdas {n_lambdas} - Number of data points {n_datapoints} - Number of polynomials {degree}.png'
    elif regression_method == "OLS":
        x_axis_label = "Polynomial degree"
        #text = f'{regression_method} - MSE for Different {x_axis_label} {noise_text} Noise\nNumber of data points: {n_datapoints}\nNumber of polynomials {degree}'
        filename = f'{regression_method} - MSE for Different {x_axis_label} {noise_text} Noise - Number of data points {n_datapoints} - Number of polynomials {degree}.png'

    # removed title from plot, but keep code in case needed later
    #plt.title(text)  
    plt.plot(x_axis_data, mse_train, label='MSE train')
    plt.plot(x_axis_data, mse_test, label='MSE test')
    plt.xlabel(x_axis_label)
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()



def plot_r2(regression_method, degree, n_datapoints, x_axis_data, r2_train, r2_test, noise=False):
    """
    Plots the R2 Score for different polynomial degrees.
    
    Returns
    -------
    Saves and shows a plot of R2 Score for training and test sets.

    Parameters
    ----------
    regression_method : string
        Type of regression for plotting
    
    degree : int
        number of polynomials in regression

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
        noise_text = "with"
    else:
        noise_text = "without"
    
    if regression_method == "Ridge":
        x_axis_label = f"Lambdas"
        n_lambdas = len(x_axis_data)
        #text = f'{regression_method} - R2 for Different {x_axis_label} {noise_text} Noise\nNumber of lambdas {n_lambdas}\nNumber of data points: {n_datapoints}\nNumber of polynomials {degree}'
        filename = f'{regression_method} - R2 for Different {x_axis_label} {noise_text} Noise - Number of lambdas {n_lambdas} - Number of data points {n_datapoints} - Number of polynomials {degree}.png'
    elif regression_method == "OLS":
        x_axis_label = "Polynomial degree"
        #text = f'{regression_method} - R2 for Different {x_axis_label} {noise_text} Noise\nNumber of data points: {n_datapoints}\nNumber of polynomials {degree}'
        filename = f'{regression_method} - R2 for Different {x_axis_label} {noise_text} Noise - Number of data points {n_datapoints} - Number of polynomials {degree}.png'

    # removed title from plot, but keep code in case needed later
    #plt.title(text)  
    plt.plot(x_axis_data, r2_train, label='R2 train')
    plt.plot(x_axis_data, r2_test, label='R2 test')
    plt.xlabel(x_axis_label)
    plt.ylabel('R2')
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

    """
    if noise:
        plt.title(f'{regression_method} - R2 Score for Different {x_axis_label} with Noise\nNumber of data points: {n_datapoints}')
        filename = f'{regression_method} - R2 for Different {x_axis_label} with Noise - Number of data points {n_datapoints}.png'
    else:
        plt.title(f'{regression_method} - R2 Score for Different {x_axis_label} without Noise\nNumber of data points: {n_datapoints}')
        filename = f'{regression_method} - R2 for Different {x_axis_label} without Noise - Number of data points {n_datapoints}.png'
    plt.plot(x_axis_data, r2_train, label='R2 train')
    plt.plot(x_axis_data, r2_test, label='R2 test')
    plt.xlabel(x_axis_label)
    plt.ylabel('R2 Score')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()
    """

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
    thetas = list() # thetas for polynomial degrees for plotting

    for degree in range(1, p+1):
        polynomial_degree.append(degree)

        # Extract the relevant columns from design matrix for the current degree
        X_train_sliced = X_train[:, :degree+1] 
        X_test_sliced = X_test[:, :degree+1]
        
        # OLS Regression
        theta_OLS = OLS_parameters(X_train_sliced, y_train)
        y_tilde_train = X_train_sliced @ theta_OLS
        y_tilde_test = X_test_sliced @ theta_OLS
        thetas.append(theta_OLS)

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
            print(f"Polynomial degree: {degree}, Coef: {model.coef_}, intercept: {model.intercept_}")
            print('\n') # just to add line shift between different degrees in output
    
    return polynomial_degree, mse_train, mse_test, r2_train, r2_test, thetas


def explore_lambda(X_train, X_test, y_train, y_test, lambdas, verbose=False):
    """
    Explores the effect of polynomial degree on MSE and R2 for
    both training and test datasets using Ridge regression.

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
    
    lambdas : numpy array shape (n)
        lambda values
    
    verbose : Bool
        Include verbose output from function, default set to false
   
    """

    mse_train = []
    mse_test = []
    r2_train  = []
    r2_test = []
    theta_ridge_list = []

    for l in lambdas:
        # Apply ridge regression
        theta_ridge = Ridge_parameters(X_train, y_train,l)
        theta_ridge_list.append(theta_ridge)
        y_tilde_train = X_train @ theta_ridge
        y_tilde_test = X_test @ theta_ridge

        if verbose: print(f"Ridge: Lambda: {l}, Coef: {theta_ridge}")


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
        if verbose: print(f"Lambda: {l}, R2_train_ridge: {r2_train_ridge}, R2_test_ridge: {r2_test_ridge} \n")

    return mse_train, mse_test, r2_train, r2_test, theta_ridge_list



def lasso_grid_search(X_train, X_test, y_train, y_test, lambdas, learning_rate, tol, max_iter, fit_intercept, verbose=False):
    """
    Grid search og lamda and eta values for finding optimal parameters for Lasso
    regression, where lowest mse is defined as optimal criterion

    Returns
    -------
    best_lasso: list
        list with dictionaries for lambda, learning rate, mse, coef and intercept for all
        analyzed variants of lambda and eta values
    
    mse_train: float
        value for MSE based on training data
    
    mse_test: float
        value for MSE based on test data
    
    r2_train: float
        value for R2 based on training data
    
    r2_test: float
        value for R2 based on test data
    
    mse_values: list
        list with mse values for combinations of lambda and eta
        
    Parameters
    ----------
    X_train : numpy array shape (n,f)
        Feature matrix for the train data, where n is the number
        of data points and f is the number of features.
    
    X_test : numpy array shape (n,f)
        Feature matrix for the test data, where n is the number
        of data points and f is the number of features.

    y_train : numpy array shape (n)
        Y values of the train data set. 

    y_test : numpy array shape (n)
        Y values of the test data set. 
    
    lambda_ : int
        regularization
    
    learning_rate : float
        gradient descent parameter
    
    tol: float
        tolerance for convergence stopping criteria
    
    max_iter : int
        number of iterations

    use_intercept : Bool
        Bool to determine if intercept should be included or not in regression:
        False : no intercept 
        True : include intercept
        
    verbose : Bool
        Include verbose output from function, default set to false
    """

    lasso_results = []
    mse_train = []
    mse_test = []
    r2_train = []
    r2_test = []

    for lambda_ in lambdas:
        for lr in learning_rate:
            coef, intercept = lasso_gradient_descent(X_train, y_train, lambda_, lr, tol, max_iter, fit_intercept)
            y_tilde_test = X_test @ coef  + intercept
            mse = MSE(y_test, y_tilde_test)
            r2 = R2(y_test, y_tilde_test)
            lasso_results.append({
                'lambda': lambda_,
                'learning_rate': lr,
                'mse': mse,
                'r2': r2,
                'coef': coef,
                'intercept': intercept
            })

    best_lasso = min(lasso_results, key=lambda r: r['mse'])

    
    # Extract all MSE values from the list
    mse_values = np.array([entry['mse'] for entry in lasso_results if 'mse' in entry])
    r2_values = np.array([entry['r2'] for entry in lasso_results if 'r2' in entry])


    y_tilde_train = X_train @ best_lasso['coef'] + best_lasso['intercept']
    y_tilde_test = X_test @ best_lasso['coef'] + best_lasso['intercept']

    # Calculate MSE for training and test data - best value 
    mse_train_lasso = MSE(y_train, y_tilde_train)
    mse_test_lasso = MSE(y_test, y_tilde_test)
    mse_train.append(mse_train_lasso)
    mse_test.append(mse_test_lasso)
    if verbose: print(f"Lasso: Lambda: {lambda_}, MSE_train_lasso: {mse_train_lasso}, MSE_test_lasso: {mse_test_lasso}")
    
    # Calculate R2 for training and test data - best value 
    r2_train_lasso = R2(y_train, y_tilde_train)
    r2_test_lasso = R2(y_test, y_tilde_test)
    r2_train.append(r2_train_lasso)
    r2_test.append(r2_test_lasso)
    if verbose: print(f"Lasso: Lambda: {lambda_}, R2_train_lasso: {r2_train_lasso}, R2_test_lsso: {r2_test_lasso}")

    if verbose:
        print(f"Lasso own implementation best result: Lambda: {best_lasso['lambda']}, Learning rate: {best_lasso['learning_rate']}, MSE: {best_lasso['mse']}")
        print(f"Lasso coef: {best_lasso['coef']}, intercept: {best_lasso['intercept']}")

    return best_lasso, mse_train, mse_test, r2_train, r2_test, mse_values, r2_values




def plot_heatmap_lasso(mse_or_r2, mse_array, lambdas, etas, degree, n_datapoints, n_iter):
    """
    Not used, kept for documentation. See function heatmap_variable_colwidth

    Plotting of heatmap from mse values from lasso_grid_search
    Depends on number of lambda values to explore and learning rate (etas)

    Some minor help code contributions form Copilot for labeling axis correctly.

    Returns
    -------
    None
        
    Parameters
    ----------
    mse_or_r2 : string
        type of metric

    mse_array : numpy array shape (n)
        array with mse values 

    lambdas : array
        lambda values to explore

    etas : list
        eta (learning rate) values explored

    degree : int
        polynomial degree

    n_datapoints : int
        number of datapoints in regression

    n_iter : int
        number of iterations
    """
    lambda_n = len(lambdas)
    mse_matrix = np.array(mse_array).reshape((lambda_n, len(etas)))

    lambdas = np.asarray(lambdas)
    etas = np.asarray(etas)

    L = len(lambdas)
    E = len(etas)
    mse_matrix = np.array(mse_array).reshape((L, E))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    # Keep imshow in index space (so your text annotations still land correctly)
    im = ax.imshow(mse_matrix, aspect='auto', origin='upper')

    for i in range(0, (L)):
        for j in range(E):
            #text = ax.text(j, i, f"{mse_matrix[i, j]:.2e}",
            text = ax.text(j, i, f"{mse_matrix[i, j]:.3f}",
                        ha="center", va="center", color="black", fontsize = 12)

    

    # X-axis: all eta values
    ax.set_xticks(range(E))
    ax.set_xticklabels([f"{eta:.3f}" for eta in etas], rotation=45)

    # Y-axis: all lambda values
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"{lmbd:.3f}" for lmbd in lambdas])


    # After imshow and before plt.show()
    ax.set_xticks(range(E))
    ax.set_xticklabels([f"{eta:.3f}" for eta in etas], rotation=45)

    ax.set_yticks(range(L))
    ax.set_yticklabels([f"{lmbd:.3f}" for lmbd in lambdas])

    ax.set_ylabel("Lambdas", fontsize=12)
    ax.set_xlabel("Learning rate", fontsize=12)
    ax.set_title(f"Heatmap Lasso regression with {mse_or_r2} - Number of lambdas {lambda_n} \nNumber of learning rate {len(etas)}\npolynomial degree {degree}\nNumber of datapoints {n_datapoints}\nNumber of iterations {n_iter}", fontsize=12)
    plt.colorbar(im)
    plt.savefig(f'Heatmap Lasso regression with {mse_or_r2} - Number of lambdas {lambda_n} number of learning rate {len(etas)} - polynomial degree {degree} - datapoints {n_datapoints} - Number of iterations {n_iter}.png', bbox_inches='tight')
    plt.show()
    plt.close()




def heatmap_variable_colwidth(mse_array, lambdas, etas, mse_or_r2, degree, n_datapoints, n_iter):
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    """
    Plotting of heatmap from mse values from lasso_grid_search
    Depends on number of lambda values to explore and learning rate (etas)
    
    Draw a heatmap where each column width (and optionally row height) fits the text.
    Based on function plot_heatmap_lasso where Microsoft Copilot were asked Question
        ####
        QUESTION: can i set size of column in heatmap to size of text?
            --> provided code to Copilot    
                # Keep imshow in index space (so your text annotations still land correctly)
                im = ax.imshow(mse_matrix, aspect='auto', origin='upper')

                for i in range(0, (L)):
                    for j in range(E):
                        #text = ax.text(j, i, f"{mse_matrix[i, j]:.2e}",
                        text = ax.text(j, i, f"{mse_matrix[i, j]:.3f}",
                                    ha="center", va="center", color="black", fontsize = 12)
    Minor modifications to provided code from Copilot
                                    

    Returns
    -------
    None
        
    Parameters
    ----------
    mse_array : numpy array shape (n)
        array with mse values 

    lambdas : array
        lambda values to explore

    etas : list
        eta (learning rate) values explored
    
    mse_or_r2 : string
        type of metric
        used in filename
  
    degree : int
        polynomial degree
        used in filename
    
    n_iter : int
        number of iterations
        used in filename
    """

    fmt="{:.3f}"
    fontsize=12
    cmap="viridis"
    pad=1.10
    scale_rows=True
    text_color="white"

    mse_matrix = np.array(mse_array).reshape((len(lambdas), len(etas)))

    L, E = mse_matrix.shape
    labels = np.array([[fmt.format(mse_matrix[i, j]) for j in range(E)] for i in range(L)], dtype=object)

    fp = FontProperties(size=fontsize)

    # Measure text sizes (units are in font units; proportionality is all we need)
    widths = np.zeros_like(mse_matrix, dtype=float)
    heights = np.zeros_like(mse_matrix, dtype=float)
    for i in range(L):
        for j in range(E):
            tp = TextPath((0, 0), labels[i, j], prop=fp)
            bb = tp.get_extents()
            widths[i, j] = bb.width
            heights[i, j] = bb.height

    # Column widths: use the widest label in each column
    col_w = pad * widths.max(axis=0)

    # Row heights: either scale to tallest text in each row, or keep uniform
    if scale_rows:
        row_h = pad * heights.max(axis=1)
    else:
        row_h = np.full(L, pad * heights.max())  # same height for all rows

    # Build grid edges (non-uniform)
    x_edges = np.concatenate(([0], np.cumsum(col_w)))   # length E+1
    y_edges = np.concatenate(([0], np.cumsum(row_h)))   # length L+1

    # Create the plot
    fig, ax = plt.subplots(constrained_layout=True)
    X, Y = np.meshgrid(x_edges, y_edges)

    # pcolormesh expects M with shape (Ny, Nx) == (L, E)
    # Default origin is lower; we invert y to resemble 'origin="upper"' behavior.
    mesh = ax.pcolormesh(X, Y, mse_matrix, cmap=cmap, shading='flat')

    # Compute centers for annotations and ticks
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Put the labels in the center of each cell
    for i in range(L):
        for j in range(E):
            ax.text(x_centers[j], y_centers[i], labels[i, j],
                    ha='center', va='center', color=text_color, fontsize=fontsize)

    # Make row 0 appear at the top (like origin='upper' with imshow)
    ax.set_ylim(y_edges[-1], y_edges[0])
    ax.set_aspect('auto')

    # Optional: ticks at centers
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)
    ax.set_ylabel("Lambdas", fontsize=12)
    ax.set_xlabel("Learning rate", fontsize=12)
    ax.set_xticklabels([f"{eta:.3f}" for eta in etas], fontsize=12)#, rotation=45)
    #ax.set_yticklabels([f"{lmbd:.3f}" for lmbd in lambdas], fontsize=12)
    ax.set_yticklabels([f"{lmbd:.1e}" for lmbd in lambdas], fontsize=12) # scientific notation
    

    
    cbar = plt.colorbar(mesh, ax=ax)  # Link colorbar to your pcolormesh
    cbar.ax.tick_params(labelsize=fontsize)  # Set tick font size
    cbar.set_label('MSE', fontsize=fontsize)  # Optional: add label

    plt.savefig(f'Heatmap Lasso regression with {mse_or_r2} - Number of lambdas {len(lambdas)} number of learning rate {len(etas)} - polynomial degree {degree} - datapoints {n_datapoints} - Number of iterations {n_iter}.png', bbox_inches='tight')
    plt.show()
    plt.close()





def plot_theta_by_polynomials(thetas, degree, n_datapoints):
    """
    Plotting thetas as function of polynomial degree

    Returns
    -------
    None
        
    Parameters
    ----------
    thetas: list
        with values of theta for each polynomial degree analysed

    degree : int
        Polynomial degree used in analysis

    n_datapoints : int
        number of datapoints in regression
    """
    
    # ensure integer values at x-axis
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for i, theta in enumerate(thetas):
        plt.plot(range(len(theta)), theta, label=f'Degree {i + 1}', alpha=0.7)
    
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Theta Value', fontsize=12)
    # removed title from plotting, code kept in case needed later
    #plt.title(f'Theta values by polynomial degree {degree} - OLS regression\nDatapoints {n_datapoints}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  
    plt.savefig(f'Theta values by polynomial degree {degree} - OLS regression - Datapoints {n_datapoints}', bbox_inches='tight')
    plt.show()
    plt.close()



def plot_theta_by_polynomials_comparison(thetas, degree, n_datapoints, regressions):
    """
    Plotting thetas as function of polynomial degree

    Returns
    -------
    None
        
    Parameters
    ----------
    thetas: list
        with values of theta for each polynomial degree analysed

    degree : int
        Polynomial degree used in analysis

    regressions : list
        string values for legend in plot    
    """
    
    # ensure integer values at x-axis
    import matplotlib.ticker as ticker
    # Ensure integer values at x-axis
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Plot each theta series
    for i, theta in enumerate(thetas):
        plt.plot(range(1, len(theta) + 1), theta, label=regressions[i], alpha=0.7)

    # Axis labels
    plt.xlabel('Degree', fontsize=12)
    plt.ylabel('Theta Value', fontsize=12)

    # Set x-axis ticks from 1 to 15
    plt.xticks(range(1, 16))

    # Legend and layout
    plt.legend()
    plt.tight_layout(rect=[0, 0, 0.85, 1])  

    # Save and show plot
    plt.savefig(f'Theta values by polynomial degree {degree} datapoints {n_datapoints} {regressions}', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_xy_xynoise_ypredicted(x, y, x_train, y_train, y_predicted_rescaled, x_test, n_datapoints, regression_method, poly_degree, noise, n_lambdas, lambda_value, eta, n_iter):

    """
    Plot x, y, x_train, y_train and x_test with predicted y_values at original scale.
    Plots are shown with variables as n_datapoints, regression_method, poly_degree and noise

    Returns
    -------
    Saved figure
        
    Parameters
    ----------
    x : numpy array shape (n)
        x values of function without noise
    
    y : numpy array shape (n)
        y values of function without noise
    
    x_noise : numpy array shape (n)
        x values of function with noise
    
    y_noise : numpy array shape (n)
        y values of function with noise

    y_predicted_rescaled : numpy array shape (n)
        Predicted y values at original scale

    x_test : numpy array shape (n)
        x values of test dataset

    n_datapoints : int
        variable used for plotting of n_datapoints in regression

    regression_method : string
        String for regression method in plot
    
    n_datapoints : int
        variable used for plotting of n_datapoints in regression

    poly_degree : int
        variable used for plotting of poly_degree in regression

    noise : bool
        decide if noise or without noise should be in the plot

    lambdas_n : int
        number of lamda values explored, regularization parameter

    lambda_value : float
        value of lambda used for plotting Ridge with the number in the title

    etas : list
        eta (learning rate) values explored
        only for Lasso regression

    n_iter : int
        number of iterations
    """
    

    plt.figure(figsize=(12, 8))
    plt.plot(x,y, color='blue', label='Runges function')
    size_scatter_train = 6
    size_scatter_predicted = 16
    plt.scatter(x_train, y_train, marker='o', s=size_scatter_train, color='green', label='Runges function - training data')
    plt.scatter(x_test, y_predicted_rescaled, s=size_scatter_predicted, marker='o', color='red', label='Runges function - predicted and rescaled')

    if noise:
        noise_text = "with"
    else:
        noise_text = "without"

    # writing information to plot title removed, but kept in code if needed later
    if regression_method == "OLS":
        #text = f'Runges function with {regression_method} regression {noise_text}\nNumber of data points: {n_datapoints}\nPolynomial degree: {poly_degree}'
        filename = f'Runges function with {regression_method} regression {noise_text} noise - Number of data points {n_datapoints} Polynomial degree - {poly_degree}.png'
    elif regression_method == "Ridge":
        #text = f'Runges function with {regression_method} regression {noise_text}\nNumber of data points: {n_datapoints}\nPolynomial degree: {poly_degree}\nLambda value: {lambda_value}'
        filename = f'Runges function with {regression_method} regression {noise_text} noise - Number of data points {n_datapoints} Polynomial degree - {poly_degree} - Lambda value {lambda_value}.png'
    elif regression_method == "Ridge-gradient":
        #text = f'Runges function with {regression_method} regression {noise_text}\nNumber of data points: {n_datapoints}\nPolynomial degree: {poly_degree}\nLambda value: {lambda_value}\nNumber of iterations: {n_iter}'
        filename = f'Runges function with {regression_method} regression {noise_text} noise - Number of data points {n_datapoints} Polynomial degree - {poly_degree} - Lambda value {lambda_value} - Number of iterations {n_iter}.png'
    elif regression_method == "Lasso":
        #text = f'Runges function with {regression_method} regression {noise_text}\nNumber of data points: {n_datapoints}\nPolynomial degree: {poly_degree}\nNumber of lambdas: {n_lambdas}\n Number of learning rate: {len(eta)}\nNumber of iterations: {n_iter}'
        filename = f'Runges function with {regression_method} regression {noise_text} noise - Number of data points {n_datapoints} Polynomial degree - {poly_degree} - Number of lambdas {n_lambdas} - Number of learning rate{len(eta)} - Number of iterations {n_iter}.png'
    else:
        raise ValueError(f"Unknown regression method: {regression_method}")
    
    #plt.title(text)
    f_size =20
    plt.legend(fontsize=f_size)
    plt.xlabel('X values', fontsize=f_size)
    plt.ylabel('Y values', fontsize=f_size)
    plt.xticks(fontsize=f_size)
    plt.yticks(fontsize=f_size)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()



