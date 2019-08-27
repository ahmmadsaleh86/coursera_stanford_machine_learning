## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient
#     randInitializeWeights
#     nnCostFunction
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

def randInitializeWeights(L_in, L_out):
    # RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    # incoming connections and L_out outgoing connections
    #   RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
    #   of a layer with L_in incoming connections and L_out outgoing 
    #   connections. 
    #
    #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    #   the column row of W handles the "bias" terms
    #

    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    
    epsilone = 0.12
    W = np.random.uniform(low=-epsilone, high=epsilone, size=(L_out, L_in + 1))
    
    return W
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
    
def predict(Theta1, Theta2, X):
   
    # Setup some useful variables
    m, n = X.shape
    
    #Adding the x0 to the training set
    XOne = np.ones((m, n+1), dtype=float)
    XOne[:, 1:] = X
    
    #Apply feedforward to find the result
    hX = h(Theta1, Theta2, XOne)
    
    p = np.argmax(hX, axis=1)
    p += 1
    p[p == 11] = 10
    
    return p

def sigmoidGradient(z):
    # SIGMOIDGRADIENT returns the gradient of the sigmoid function
    # evaluated at z
    #   SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    #   evaluated at z. This should work regardless if z is a matrix or a
    #   vector. In particular, if z is a vector or matrix, you should return
    #   the gradient for each element.
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).
    g = sigmoid(z)
    
    gGradient = np.multiply(g, (1 - g))
    
    return gGradient
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
    #   X, y, lambda) computes the cost of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices. 
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
    #         computed
    
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
    
    #regularization
    Theta1_reg = Theta1[:, 1:]
    Theta2_reg = Theta2[:, 1:]
    
    J += (lambd / (2 * m)) * (np.sum(np.power(Theta1_reg, 2)) + np.sum(np.power(Theta2_reg, 2)))
    
    return J
#END

def nnGradient(nn_params,
               input_layer_size,
               hidden_layer_size,
               num_labels,
               X,
               y,
               lambd):
    # NNGradient Implements the gradient for a two layer
    # neural network which performs classification
    #   NNGradient(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the gradient of the neural network. The
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
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    # 
            
    #Adding the x0 to the training set
    A1 = np.ones((m, input_layer_size + 1), dtype=float)
    A1[:, 1:] = X
    
    #Convert y to binary presentation
    yD = np.zeros((m, num_labels), dtype=float)
    y = y.astype(int)
    for i in range(m):
        yD[i, y[i]-1] = 1
    
    #Applying feedforward propagation
    Z2 = np.matmul(A1, Theta1.T)
    A2 = np.ones(((Z2.shape)[0], (Z2.shape)[1] + 1), dtype=float)
    A2[:, 1:] = sigmoid(Z2)
    
    Z3 = np.matmul(A2, Theta2.T)
    A3 = sigmoid(Z3)
    
    #Applying back propagation propagation
    delta3 = A3 - yD
    
    delta2 = np.matmul(delta3, Theta2)
    delta2 = delta2[:, 1:]
    delta2 = np.multiply(delta2, sigmoidGradient(Z2))
    
    Theta2_grad = (1.0 / m) * np.matmul(delta3.T, A2)
    
    Theta1_grad = (1.0 / m) * np.matmul(delta2.T, A1)
    
    #regularization
    Theta1_reg = Theta1[:, 1:]
    Theta2_reg = Theta2[:, 1:]
    
    Theta1_grad[:, 1:] += (lambd / m) * Theta1_reg
    Theta2_grad[:, 1:] += (lambd / m) * Theta2_reg
    
    # Unroll gradients
    Theta1_grad_num = (Theta1_grad.shape)[0] * (Theta1_grad.shape)[1]
    Theta2_grad_num = (Theta2_grad.shape)[0] * (Theta2_grad.shape)[1]
    
    grad = np.zeros((Theta1_grad_num + Theta2_grad_num, ), dtype=float)
    grad[:Theta1_grad_num] = Theta1_grad.ravel()
    grad[Theta1_grad_num:] = Theta2_grad.ravel()
    
    return grad
#END
    
## ====================  Gradient Checking Functions ====================

def debugInitializeWeights(fan_out, fan_in):
    # DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
    # incoming connections and fan_out outgoing connections using a fixed
    # strategy, this will help you later in debugging
    #   DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
    #   of a layer with fan_in incoming connections and fan_out outgoing 
    #   connections using a fix set of values
    #
    #   Note that W should be set to a matrix of size(fan_out, 1 + fan_in) as
    #   the first row of W handles the "bias" terms
    #
    
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    numElement = fan_out * (fan_in + 1)
    W = np.linspace(start=1, stop=numElement, num=numElement)
    W = np.sin(W)
    W = np.reshape(W, (fan_out, fan_in+1)) / 10
    
    return W
#END
    
def computeNumericalGradient(nn_params,
               input_layer_size,
               hidden_layer_size,
               num_labels,
               X,
               y,
               lambd):
    # COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    # and gives us a numerical estimate of the gradient.
    #   COMPUTENUMERICALGRADIENT(nn_params, input_layer_size, hidden_layer_size,
    #   num_labels, X, y, lambd) computes the numerical gradient of the
    #   function J around theta.
    #
    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    #
    
    numgrad = np.zeros(nn_params.shape, dtype=float)
    perturb = np.zeros(nn_params.shape, dtype=float)
    e = 1e-4
    
    for i in range((nn_params.shape)[0]):
        perturb[i] = e
        loss1 = nnCostFunction(nn_params - perturb,
               input_layer_size,
               hidden_layer_size,
               num_labels,
               X,
               y,
               lambd)
        
        loss2 = nnCostFunction(nn_params + perturb,
               input_layer_size,
               hidden_layer_size,
               num_labels,
               X,
               y,
               lambd)
        
        numgrad[i] = (loss2 - loss1) / (2 * e)
        
        perturb[i] = 0
    
    return numgrad
#END


def checkNNGradients(lambd = 0):
    # CHECKNNGRADIENTS Creates a small neural network to check the
    # backpropagation gradients
    #   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using computeNumericalGradient). These two gradient computations should
    #   result in very similar values.
    #
    
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    
    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size-1)
    y = 1 + np.linspace(start=1, stop=m, num=m) % num_labels
    
    # Unroll parameters 
    nn_params = np.zeros((hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1), ), dtype=float)
    nn_params[:hidden_layer_size * (input_layer_size + 1)] = Theta1.ravel()
    nn_params[hidden_layer_size * (input_layer_size + 1):] = Theta2.ravel()
    
    cost = nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X,
                   y,
                   lambd)  
    grad = nnGradient(nn_params,
               input_layer_size,
               hidden_layer_size,
               num_labels,
               X,
               y,
               lambd)
    numgrad = computeNumericalGradient(nn_params,
               input_layer_size,
               hidden_layer_size,
               num_labels,
               X,
               y,
               lambd)
    
    # Visually examine the two gradient computations.
    print(numgrad)
    print(grad)
    
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    
    print('If your backpropagation implementation is correct, then \n',
             'the relative difference will be small (less than 1e-9). \n',
             '\nRelative Difference: ', diff)
    return numgrad
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

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
lambd = 1

J = nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X,
                   y,
                   lambd)

print('Cost at parameters (loaded from ex4weights): ', J)
print('(this value should be about 0.383770)')


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))

print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ')
print(g)

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.zeros((hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1), ), dtype=float)
initial_nn_params[:hidden_layer_size * (input_layer_size + 1)] = initial_Theta1.ravel()
initial_nn_params[hidden_layer_size * (input_layer_size + 1):] = initial_Theta2.ravel()

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
#print('\nChecking Backpropagation... \n');

#  Check gradients by running checkNNGradients
#checkNNGradients()

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lambd = 3
checkNNGradients(lambd)

# Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params,
                          input_layer_size, 
                          hidden_layer_size,
                          num_labels,
                          X,
                          y,
                          lambd)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 3): ', debug_J)
print('(this value should be about 0.576051)\n\n')

## =================== Part 9: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network. Recall that the
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
import scipy.optimize as op

print('\nTraining Neural Network... \n')


#  You should also try different values of lambd
lambd = 3

nn_params = op.fmin_ncg(f=nnCostFunction,
                        x0=initial_nn_params,
                        fprime=nnGradient,
                        maxiter=50,
                        args=(input_layer_size, hidden_layer_size, num_labels, X, y, lambd))

cost = nnCostFunction(nn_params, 
                      input_layer_size,
                      hidden_layer_size,
                      num_labels,
                      X,
                      y,
                      lambd)

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1))

## ================= Part 10: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:, 1:])

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

acc = (np.sum(pred == y) * 100.0)/m

print('\nTraining Set Accuracy: ', acc)


