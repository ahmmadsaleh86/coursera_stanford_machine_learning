## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid
#     costFunction
#     predict
#     costFunctionReg
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ====================  Functions ====================
def plotData(X, y):
    # PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.    
    y0 = (np.where(y==0))[0]
    y1 = (np.where(y==1))[0]
    
    plt.scatter(X[y0, 0], X[y0, 1], color="yellow", marker='x')
    plt.scatter(X[y1, 0], X[y1, 1], color="blue")
    
    plt.title('Scatter plot of training data')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    
    plt.show()
#END
    
def sigmoid(z):
    # SIGMOID Compute sigmoid functoon
    #   g = SIGMOID(z) computes the sigmoid of z.
    g = 1 / (1 + np.exp(-z))
    return g
#END
    
def costFunction(theta, X, y):
    # COSTFUNCTION Compute cost for logistic regression
    #   J = COSTFUNCTION(X, y, theta) computes the cost of using theta as the
    #   parameter for logistic regression and.
    
    #np.matmul is a matrix multiplication
    #np.multiply is a matrix element wise multiplation
    
    # Initialize some useful values
    m = len(y) # number of training examples
    hX = sigmoid(np.matmul(X, theta))
    
    J = (1.0 / m) * np.sum(np.multiply(-y, np.log(hX)) - np.multiply((1 - y), np.log(1 - hX)))
    return J
#END
    
def gradient(theta, X, y):
    # Gradient Compute gradient for logistic regression
    #  Gradient(X, y, theta) computes the gradient of the cost w.r.t. to the parameters.    
    
    # Initialize some useful values
    m = len(y) # number of training examples
    hX = sigmoid(np.matmul(X, theta))
    
    #grad = (1 / m) * (h_theta - y)' * X
    delta = (1.0/m) * (np.matmul(X.T, (hX - y))).flatten()
    return delta
#END



## ==================== Part 0: Loading Data ====================
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

import csv
import numpy as np

dataTrack = open('ex2data1.txt')
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

print('Plotting data with blue . indicating (y = 1) examples and yellow x indicating (y = 0) examples.\n')

import matplotlib.pyplot as plt
plotData(X, y)

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression.

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
temp = X
X = np.ones((m, n+1), dtype=float)
X[:, 1:] = temp

# Initialize fitting parameters
initial_theta = np.zeros((n+1,), dtype=float)

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost at initial theta (zeros): ', cost)
print('Gradient at initial theta (zeros): ')
print(grad)
