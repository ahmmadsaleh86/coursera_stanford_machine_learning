## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
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
    
def h(Theta1, Theta2, XOne):
    Z2 = np.matmul(XOne, Theta1.T)
    A2 = np.ones(((Z2.shape)[0], (Z2.shape)[1] + 1), dtype=float)
    A2[:, 1:] = sigmoid(Z2)
    
    Z3 = np.matmul(A2, Theta2.T)
    A3 = sigmoid(Z3)
    
    return A3
#END

   
def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X,
                   y,
                   lambd):
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices. 
    # 
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))
    
    # Setup some useful variables
    m = (X.shape)[0]    
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    
    #Adding the x0 to the training set
    XOne = np.ones((m, input_layer_size + 1), dtype=float)
    XOne[:, 1:] = X
    
    #Applying feedforward propagation
    hX = h(Theta1, Theta2, XOne)
    
    
    #Convert y to binary presentation
    yD = np.zeros((m, num_labels), dtype=float)
    y = y.astype(int)
    for i in range(m):
        yD[i, y[i]-1] = 1
    
    J = (1.0 / m) * (np.sum(np.multiply(-yD, np.log(hX)) - np.multiply((1 - yD), np.log(1 - hX))))
    
    return J
"""
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%



J = (1/m) * sum ( sum ( (-y_new) .* log(h_theta) - (1-y_new) .* log(1-h_theta) ));

% Note we should not regularize the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix.
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% Regularization
Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

% Regularized cost function
J = J + Reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Back propagation
for t=1:m

    % Step 1
	a1 = X(t,:); % X already have a bias Line 44 (1*401)
    a1 = a1'; % (401*1)
	z2 = Theta1 * a1; % (25*401)*(401*1)
	a2 = sigmoid(z2); % (25*1)
    
    a2 = [1 ; a2]; % adding a bias (26*1)
	z3 = Theta2 * a2; % (10*26)*(26*1)
	a3 = sigmoid(z3); % final activation layer a3 == h(theta) (10*1)
    
    % Step 2
	delta_3 = a3 - y_new(:,t); % (10*1)
	
    z2=[1; z2]; % bias (26*1)
    % Step 3
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % ((26*10)*(10*1))=(26*1)

    % Step 4
	delta_2 = delta_2(2:end); % skipping sigma2(0) (25*1)

	Theta2_grad = Theta2_grad + delta_3 * a2'; % (10*1)*(1*26)
	Theta1_grad = Theta1_grad + delta_2 * a1'; % (25*1)*(1*401)
    
end;

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% Regularization

% Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m; % for j = 0
% 
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
% 
% Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m; % for j = 0
% 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
"""


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

mat = scipy.io.loadmat('ex4data1.mat')

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



## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...\n')

#Load the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('ex4weights.mat')
Theta1 = mat.get('Theta1')
Theta2 = mat.get('Theta2')

# Unroll parameters 
nn_params = np.zeros((hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1), ), dtype=float)
nn_params[:hidden_layer_size * (input_layer_size + 1)] = Theta1.ravel()
nn_params[hidden_layer_size * (input_layer_size + 1):] = Theta2.ravel()

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#

print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
lambd = 0

J = nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X,
                   y,
                   lambd)

print('Cost at parameters (loaded from ex4weights):  ', J)
print('(this value should be about 0.287629)')

