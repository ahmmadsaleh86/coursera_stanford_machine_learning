## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction (logistic regression cost function)
#     oneVsAll
#     predictOneVsAll
#     predict
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ====================  Functions ====================
def displayData(X, example_width = None):
    # DISPLAYDATA Display 2D data in a nice grid
    #   DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the 
    #   displayed array if requested.
    
    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(round(math.sqrt((X.shape)[1]), 0))
    
    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)
    
    # Compute number of items to display
    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m / display_rows)
    
    # Between images padding
    pad = 1
    
    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)), dtype=float)
    
    # Copy each example into a patch on the display array
    curr_ex = 1
    for j  in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m: 
                break
            
            # Copy the patch
            		            
            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            
            tmp1 = np.linspace(0, example_height, num=example_height+1)
            tmp1 = pad + j * (example_height + pad) + tmp1.astype(int)
            
            tmp2 = np.linspace(0, example_width, num=example_width+1)
            tmp2 = pad + i * (example_width + pad) + tmp2.astype(int)
            #print(pad + (j) * (example_height + pad) + tmp1.astype(int))
            #print(pad + (i) * (example_width + pad) + tmp2.astype(int))
            
            display_array[tmp1[0]:tmp1[example_height], tmp2[0]:tmp2[example_width]] = np.reshape(X[curr_ex, :], (example_height, example_width)) / max_val
            curr_ex = curr_ex + 1
        
        if curr_ex >= m: 
            break 
        
    imgplot = plt.imshow(display_array.T, cmap='gray')
    
    return display_array
#END
    
def sigmoid(z):
    # SIGMOID Compute sigmoid functoon
    #   g = SIGMOID(z) computes the sigmoid of z.
    g = 1 / (1 + np.exp(-z))
    return g
#END

def lrCostFunctionReg(theta, X, y, lambd):
    #lrCostFunctionReg Compute cost for logistic regression with regularization
    #   lrCostFunctionReg(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression 
    
    # Initialize some useful values
    m = len(y) # number of training examples
    hX = sigmoid(np.matmul(X, theta))
    
    J = (1.0 / m) * np.sum(np.multiply(-y, np.log(hX)) - np.multiply((1 - y), np.log(1 - hX))) + (lambd / (2 * m)) * (np.sum(theta[1:]**2))
    return J
#END

def lrGradientReg(theta, X, y, lambd):
    #lrGradientReg Compute gradient for logistic regression with regularization
    #   lrGradientReg(theta, X, y, lambda) computes 
    #   the gradient of the cost w.r.t. to the parameters. 
    
    # Initialize some useful values
    m, n = X.shape
    hX = sigmoid(np.matmul(X, theta))
    thetaZero = np.zeros((n,), dtype=float)
    thetaZero[1:] = theta[1:]
    
    #grad = (1 / m) * (h_theta - y)' * X
    delta = (1.0/m) * (np.matmul(X.T, (hX - y))).flatten() + (lambd / m) * thetaZero
    
    return delta
#END
   
def oneVsAll(X, y, num_labels, lambd):
    # ONEVSALL trains multiple logistic regression classifiers and returns all
    # the classifiers in a matrix all_theta, where the i-th row of all_theta 
    # corresponds to the classifier for label i
    #   ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds 
    #   to the classifier for label i  
    
    # Some useful variables
    m, n = X.shape
    
    all_theta = np.zeros((num_labels, n+1), dtype=float)
    
    XOne = np.ones((m, n+1), dtype=float)
    XOne[:, 1:] = X
    
    tmp = np.zeros((num_labels, ), dtype=int)
    tmp[1: ] = np.linspace(1, num_labels-1, num = num_labels-1)
    tmp[0] = 10
    
    yOnevsAll = np.zeros((m, num_labels), dtype=float)
    for i in range(m):
        yOnevsAll[i, :] = tmp == y[i]
    
    for i in range(num_labels):
        all_theta[i, :] = op.fmin_ncg(f=lrCostFunctionReg, x0=all_theta[i,:], fprime=lrGradientReg, args=(XOne, yOnevsAll[:,i], lambd))
        cost = lrCostFunctionReg(all_theta[i, :], XOne, yOnevsAll[:,i], lambd)
        print("The cost for classifier ", i, " = ", cost)
        
    return all_theta
#END
    
def predictOneVsAll(all_theta, X):
    # predictOneVsAll Predict the label for a trained one-vs-all classifier. The labels 
    # are in the range 1..K, where K = size(all_theta, 1). 
    #  PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    #  for each example in the matrix X. Note that X contains the examples in
    #  rows. all_theta is a matrix where the i-th row is a trained logistic
    #  regression theta vector for the i-th class. You should set p to a vector
    #  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    #  for 4 examples) 
    
    m, n = X.shape
    
    # You need to return the following variables correctly 
    p = np.zeros((m, ), dtype=float)
    
    # Add ones to the X data matrix
    XOne = np.ones((m, n+1), dtype=float)
    XOne[:, 1:] = X
    
    prediction = sigmoid(np.matmul(XOne, all_theta.T))
    
    for i in range(m):
        maxi = np.argmax(prediction[i,:])
        if maxi == 0:
            p[i] = 10
        else:
            p[i] = maxi
    
    return p
#END

## =========== Part 0: Setup the Parameters =============
## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

# training data stored in arrays X, y
import scipy.io
import numpy as np

mat = scipy.io.loadmat('ex3data1.mat')

X = mat.get('X')
m = len(X)

yMat = mat.get('y')
y = np.zeros((m, ), dtype=float)
y[:] = yMat[:, 0]

import math
import matplotlib.pyplot as plt

# Randomly select 100 data points to display
rand_indices = np.random.randint(m, size=100)
sel = X[rand_indices, :]

displayData(sel)

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

print('\nTraining One-vs-All Logistic Regression...\n')

lambd = 0.1

# each X row corresponds to training data digit image 20*20 pixel
# each y row contains label for the training data i.e. actual name of the
# digit

import scipy.optimize as op

all_theta = oneVsAll(X, y, num_labels, lambd)

## ================ Part 3: Predict for One-Vs-All ================
#  After ...
pred = predictOneVsAll(all_theta, X)

#imgplot = plt.imshow((np.reshape(X[1087,:], (20,20))).T, cmap='gray')

acc = (np.sum(pred == y) * 100.0)/m

print('\nTraining Set Accuracy: ', acc)





