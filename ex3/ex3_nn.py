## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     predict.m
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

## =========== Part 0: Setup the Parameters =============
## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
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

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('ex3weights.mat')
theta1 = mat.get('Theta1')
theta2 = mat.get('Theta2')

