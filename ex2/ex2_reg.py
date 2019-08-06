## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ====================  Functions ====================
def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with green o for the positive examples
    #   and red x for the negative examples. X is assumed to be a Mx2 matrix.    
    y0 = (np.where(y==0))[0]
    y1 = (np.where(y==1))[0]
    
    plt.scatter(X[y0, 0], X[y0, 1], color="red", marker='x')
    plt.scatter(X[y1, 0], X[y1, 1], color="green")
    
    plt.title('Scatter plot of training data')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    
    plt.show()
#END

def plotDecisionBoundary(theta, X, y):   
    # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    # the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with green o for the 
    #   positive examples and red x for the negative examples. X is assumed to be 
    #   MxN, N>3 matrix, where the first column is all-ones
    y0 = (np.where(y==0))[0]
    y1 = (np.where(y==1))[0]
    
    plt.scatter(X[y0, 1], X[y0, 2], color="red", marker='x')
    plt.scatter(X[y1, 1], X[y1, 2], color="green")
    
    #
    u = np.linspace(-1, 1.5, num = 50)
    v = np.linspace(-1, 1.5, num = 50)
    
    z = np.zeros((len(u), len(v)), dtype=float)
        
    # Evaluate z = theta*x over the grid
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.matmul(mapFeature(np.array([u[i]]), np.array([v[j]])), theta)
    z = z.T # important to transpose z before calling contour
    
    plt.contour(u, v, z, levels=0)
    
    plt.title('Training data with decision boundary')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    
    plt.show()
#END


def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of 
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #
    
    degree = 6
    num = (degree+1) * (degree+2) / 2
    
    features = np.ones((len(X1), int(num)), dtype=float)
    
    count = 1
    for i in range(1, degree+1):
      for j in range(i+1):
        features[:, count] = np.multiply(X1**(i-j), X2**(j))
        count += 1
        
    return features
#END
    
def sigmoid(z):
    # SIGMOID Compute sigmoid functoon
    #   g = SIGMOID(z) computes the sigmoid of z.
    g = 1 / (1 + np.exp(-z))
    return g
#END

def costFunctionReg(theta, X, y, lambd):
    # COSTFUNCTIONREG Compute cost for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression
    
    # Initialize some useful values
    m = len(y) # number of training examples
    hX = sigmoid(np.matmul(X, theta))
    
    J = (1.0 / m) * np.sum(np.multiply(-y, np.log(hX)) - np.multiply((1 - y), np.log(1 - hX))) + (lambd / (2 * m)) * (np.sum(theta[1:]**2))
    return J
#END

def gradientReg(theta, X, y, lambd):
    # gradient Compute gradient for logistic regression with regularization
    #   gradient(theta, X, y, lambda) computes the gradient of the cost w.r.t. to the parameters.   
    
    # Initialize some useful values
    m, n = X.shape
    hX = sigmoid(np.matmul(X, theta))
    thetaZero = np.zeros((n,), dtype=float)
    thetaZero[1:] = theta[1:]
    
    #grad = (1 / m) * (h_theta - y)' * X
    delta = (1.0/m) * (np.matmul(X.T, (hX - y))).flatten() + (lambd / m) * thetaZero
    
    return delta
#END

def predict(theta, X):
    # PREDICT Predict whether the label is 0 or 1 using learned logistic 
    # regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a 
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    p = sigmoid(np.matmul(X, theta))
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    return p
#END    


## ==================== Part 0: Loading Data ====================
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

import csv
import numpy as np

dataTrack = open('ex2data2.txt')
csvReader = csv.reader(dataTrack)
l = list(csvReader)
m = len(l)

data = np.zeros((m, 3), dtype=float)
data[:] = l

X = data[:, :-1] #Feature 1 and Feature 2
y = data[:, -1] #Negative or Positive

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with green o indicating (y = 1) examples and red x indicating (y = 0) examples.\n')

import matplotlib.pyplot as plt
plotData(X, y)

## =========== Part 2: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:,0], X[:,1])

# Initialize fitting parameters
m, n = X.shape
initial_theta = np.zeros((n,), dtype=float)

# Set regularization parameter lambda to 1
lambd = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, lambd)
grad = gradientReg(initial_theta, X, y, lambd)

print('Cost at initial theta (zeros): ', cost)

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and 
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

import scipy.optimize as op

#  Run Newton-Conjugate-Gradient to obtain the optimal theta
theta = op.fmin_ncg(f=costFunctionReg, x0=initial_theta, fprime=gradientReg, args=(X, y, lambd))
cost = costFunctionReg(theta, X, y, lambd)

# Plot Boundary
plotDecisionBoundary(theta, X, y)


# Compute accuracy on our training set
p = predict(theta, X);

acc = (np.sum(p == y) * 100.0)/m
print('Train Accuracy: ', acc)


