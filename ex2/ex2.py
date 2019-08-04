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
