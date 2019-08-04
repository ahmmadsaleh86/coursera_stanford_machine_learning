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
    
def costFunction(X, y, theta):
    # COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.
"""   
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h_theta = sigmoid(X*theta);
J = (1 / m) * ((-y' * log(h_theta)) - (1 - y)' * log(1 - h_theta));

grad = (1 / m) * (h_theta - y)' * X;





% =============================================================

end
"""



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
#  for logistic regression. You neeed to complete the code in 
#  costFunction

#  Setup the data matrix appropriately, and add ones for the intercept term
"""
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X]; %Concatenating Column Vector 1's(Feature 0) with Column Vector X (Feature 1, Feature 2)

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
"""
