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
    #   PLOTDATA(x,y) plots the data points with green o for the positive examples
    #   and red x for the negative examples. X is assumed to be a Mx2 matrix.    
    y0 = (np.where(y==0))[0]
    y1 = (np.where(y==1))[0]
    
    plt.scatter(X[y0, 0], X[y0, 1], color="red", marker='x')
    plt.scatter(X[y1, 0], X[y1, 1], color="green")
    
    plt.title('Scatter plot of training data')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    
    plt.show()
#END
    
def plotDecisionBoundary(theta, X, y):
    # PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    # the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with green o for the 
    #   positive examples and red x for the negative examples.
    
    y0 = (np.where(y==0))[0]
    y1 = (np.where(y==1))[0]
    
    plt.scatter(X[y0, 1], X[y0, 2], color="red", marker='x')
    plt.scatter(X[y1, 1], X[y1, 2], color="green")
    
    plot_x1 = np.array([np.amin(X[:, 1]), np.amax(X[:, 1])])
    plot_x2 = (-1.0/theta[2]) * (theta[1] * plot_x1 + theta[0])
    
    plt.plot(plot_x1, plot_x2, color = 'black')
    
    plt.title('Training data with decision boundary')
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

print('Plotting data with green o indicating (y = 1) examples and red x indicating (y = 0) examples.\n')

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

## ============= Part 3: Optimizing using advance optimization problem  =============
#  In this exercise, you will use a built-in function to find the
#  optimal parameters theta.

import scipy.optimize as op

#  Run Newton-Conjugate-Gradient to obtain the optimal theta
theta = op.fmin_ncg(f=costFunction, x0=initial_theta, fprime=gradient, args=(X, y))
cost = costFunction(theta, X, y)

# Print theta to screen
print('Cost at theta found by fminunc: ', cost)
print('theta: ')
print(theta)

# Plot Boundary
plotDecisionBoundary(theta, X, y)

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

newStudent =np.array([[1, 45, 86]])
prob = sigmoid(np.matmul(newStudent, theta))
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)

# Compute accuracy on our training set
p = predict(theta, X)

acc = (np.sum(p == y) * 100.0)/m
print('Train Accuracy: ', acc)




