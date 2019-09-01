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




