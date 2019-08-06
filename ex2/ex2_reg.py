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
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.    
    y0 = (np.where(y==0))[0]
    y1 = (np.where(y==1))[0]
    
    plt.scatter(X[y0, 0], X[y0, 1], color="red", marker='x')
    plt.scatter(X[y1, 0], X[y1, 1], color="green")
    
    plt.title('Scatter plot of training data')
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

print('Plotting data with green . indicating (y = 1) examples and red x indicating (y = 0) examples.\n')

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
X = mapFeature(X[:,1], X[:,2])

# Initialize fitting parameters
m, n = X.shape
initial_theta = np.zeros((n,), dtype=float)

# Set regularization parameter lambda to 1
lambda = 1;

# Compute and display initial cost and gradient for regularized logistic
# regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
