## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction
#     learningCurve
#     validationCurve
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ====================  Functions ====================
def plotData(X, y, theta=None):
    plt.scatter(X, y, c='red', marker='x')
    
    if theta is not None:
        # Initialize some useful values
        m, n = X.shape
        XOne = np.ones((m, n+1), dtype=float)
        XOne[:, 1:] = X
        plt.plot(X, hX(theta, XOne), c='blue')
        
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
#END
    
def plotLearningCurve(error_train, error_val):
    # Initialize some useful values
    m = (error_train.shape)[0]
    
    XAxes = np.linspace(1, m, num=m)
    
    plt.plot(XAxes, error_train, c='blue')
    plt.plot(XAxes, error_val, c='green')
    
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
#END
    
    
    
def hX(theta, XOne):
    return np.matmul(XOne, theta)
#END
    
def linearRegCostFunction(theta, X, y, lambd):
    # LINEARREGCOSTFUNCTION Compute cost for regularized linear 
    # regression with multiple variables
    #   LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    #   cost of using theta as the parameter for linear regression to fit the 
    #   data points in X and y. Returns the cost in J
    
    # Initialize some useful values
    m, n = X.shape
    
    XOne = np.ones((m, n+1), dtype=float)
    XOne[:, 1:] = X
    
    J = (1 / (2 * m)) * np.sum(np.power(hX(theta, XOne) - y, 2))
    
    #Regularization
    thetaReg = theta[1:]
    J += (lambd / (2 * m)) * np.sum(np.power(thetaReg, 2))
    
    return J
#END
    
def linearRegGradient(theta, X, y, lambd):
    # LINEARREGGRADIENT Compute gradient for regularized linear 
    # regression with multiple variables
    #   LINEARREGGRADIENT(X, y, theta, lambda) computes the 
    #   gradient of using theta as the parameter for linear regression. 
    #   Returns the gradient in grad
    
    # Initialize some useful values
    m, n = X.shape
    
    XOne = np.ones((m, n+1), dtype=float)
    XOne[:, 1:] = X
    
    grad = (1 / m) * (np.matmul(XOne.T, hX(theta, XOne) - y)).ravel()
    
    #Regularization
    thetaReg = np.zeros((n+1, ), dtype=float)
    thetaReg[1:] = theta[1:]
    
    grad += (lambd / m) * thetaReg
    
    return grad
#END
    
def trainLinearReg(X, y, lambd=0):
    # TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    # regularization parameter lambd
    #   TRAINLINEARREG (X, y, lambda) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambd. Returns the
    #   trained parameters theta.
    #
    
    # Initialize some useful values
    m, n = X.shape
    
    initial_theta = np.zeros((n+1, ), dtype=float)
    
    #  Run Newton-Conjugate-Gradient to obtain the optimal theta
    theta = op.fmin_ncg(f=linearRegCostFunction,
                        x0=initial_theta,
                        fprime=linearRegGradient,
                        #maxiter=50,
                        args=(X, y, lambd))
    return theta
#END
    
def learningCurve(X, y, Xval, yval, lambd=0):
    # LEARNINGCURVE Generates the train and cross validation set errors needed 
    # to plot a learning curve
    #       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
    #       cross validation set errors for a learning curve. In particular, 
    #       it returns two vectors of the same length - error_train and 
    #       error_val. Then, error_train(i) contains the training error for
    #       i examples (and similarly for error_val(i)).
    #
    
    # Initialize some useful values
    m = (X.shape)[0]
    
    #returned variables
    error_train = np.zeros((m, ), dtype=float)
    error_val = np.zeros((m, ), dtype=float)
    
    for i in range(1, m+1):
        #training the linear regression using subset of training set(X from row zero to i-1)
        theta = trainLinearReg(X[:i, :], y[:i], lambd)
        
        #Compute the training set error using the subset only of traning set
        error_train[i-1] = linearRegCostFunction(theta, X[:i, :], y[:i], lambd)
        
        #Compute the cross validation error using the entire cross validation set
        error_val[i-1] = linearRegCostFunction(theta, Xval, yval, lambd)
        
    return error_train, error_val
#END
    
def polyFeatures(X, p):
    # Initialize some useful values
    m = (X.shape)[0]
    
    X_poly = np.ones((m, p), dtype=float)
    
    for i in range(1, p+1):
        X_poly[:, i-1] = np.power(X[:, 0], i)
    
    return X_poly
#END
    
def featureNormalize(X):
    # FEATURENORMALIZE(X) returns a normalized version of X where
    # the mean value of each feature is 0 and the standard deviation
    # is 1. This is often a good preprocessing step to do when
    # working with learning algorithms.
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = X
    X_norm = X_norm - mu
    X_norm = X_norm / sigma
    return [X_norm, mu, sigma]
#END

        
## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
import scipy.io
import numpy as np

mat = scipy.io.loadmat('ex5data1.mat')

X = mat.get('X')
Xval = mat.get('Xval')
Xtest = mat.get('Xtest')

yMat = mat.get('y')
y = np.zeros(((yMat.shape)[0], ), dtype=float)
y[:] = yMat[:, 0]

yMat = mat.get('yval')
yval = np.zeros(((yMat.shape)[0], ), dtype=float)
yval[:] = yMat[:, 0]

yMat = mat.get('ytest')
ytest = np.zeros(((yMat.shape)[0], ), dtype=float)
ytest[:] = yMat[:, 0]

# m = Number of examples
m, n = X.shape

# Plot training data
import matplotlib.pyplot as plt
plotData(X,y)


## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.ones((n+1, ), dtype=float)
lambd = 1

J = linearRegCostFunction(theta, X, y, lambd)

print('Cost at theta = [1 ; 1]: ', J)
print('(this value should be about 303.993192)\n')

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

grad = linearRegGradient(theta, X, y, lambd)

print('Gradient at theta = [1 ; 1]: ', grad)
print('\n(this value should be about [-15.303016; 598.250744])\n')


## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambd = 0
import scipy.optimize as op

lambd = 0
theta = trainLinearReg(X, y, lambd)

#  Plot fit over the data
plotData(X,y, theta)


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
#

lambd = 0
error_train, error_val = learningCurve(X, y, Xval, yval, lambd)
plotLearningCurve(error_train, error_val)

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print(i,'- train error: ', error_train[i],'----- Cross validation: ', error_val[i])

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)   # Normalize

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma

print('Normalized Training Example 1:\n')
print( X_poly[1, :])





