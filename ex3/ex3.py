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

XMat = mat.get('X')
m = len(XMat)
X = np.zeros((m, input_layer_size), dtype=float)
X[:, :] = XMat

yMat = mat.get('y')
y = np.zeros((m, ), dtype=float)
y[:] = yMat


"""
% Randomly select 100 data points to display
rand_indices = randperm(m); % Array of size 1*5000 having random no. at each positions
sel = X(rand_indices(1:100), :);

displayData(sel);
"""



