## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     warmUpExercise
#     plotData
#     gradientDescent
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

## ====================  Functions ====================
def warmUpExercise():
    return np.eye(5)

def plotData(X, Y, theta=None):
    #visualize
    plt.scatter(X[:,1], Y, color = 'red')
    if theta is not None:
        plt.plot(X[:,1], (np.matmul(X, theta)).flatten(), color = 'blue')
    plt.title('Population VS Profite')
    plt.xlabel('Populaion in $10,000s')
    plt.ylabel('profite in $10,000s')
    plt.show()

def computeCost(X, Y, theta):
    # COMPUTECOST(X, y, theta) computes the cost for linear regression 
    # using theta as the parameter for linear regression to fit the data 
    # points in X and y

    m = len(Y)
    hX = (np.matmul(X, theta)).flatten()
    J = (1/(2.0*m)) * np.sum ((hX - Y)**2)
    return J

def gradientDescent(X, Y, theta, alpha, num_iters):
    # GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    # taking num_iters gradient steps with learning rate alpha
    
    m = len(Y)
    J_history = np.zeros((num_iters, ), dtype=float)
    
    for i in range(num_iters):
        hX = (np.matmul(X, theta)).flatten()
        delta = (1.0/m) * (np.matmul(X.T, (hX - Y))).flatten()
        theta = theta - alpha * delta
        
        # Save the cost J in every iteration  
        J_history[i] = computeCost(X, Y, theta)
        
    return [theta, J_history]

## ==================== Part 1: Basic Function ====================
import numpy as np
# Complete warmUpExercise
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise())

## ======================= Part 2: Plotting =======================
import csv
print('Plotting Data ...\n')

# read comma separated data
dataTrack = open('ex1data1.txt')
csvReader = csv.reader(dataTrack)
l = list(csvReader)
m = len(l) # number of training examples
data = np.zeros([m, 2], dtype=float)
data[:] = l

X = np.ones((m, 2), dtype=float)
X[:, 1:] = data[:, :-1]
Y = data[:, -1]

#Plot Data
import matplotlib.pyplot as plt
plotData(X, Y)

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')
theta = np.zeros((2, ), dtype=float) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

# compute and display initial cost
J = computeCost(X, Y, theta)
print('initial cost is ', J)

# run gradient descent
theta, J_history = gradientDescent(X, Y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ', theta)

# Plot the linear fit
plotData(X, Y, theta)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.matmul(np.array([1, 1.8]), theta)
print('For population = 18,000, we predict a profit of ', predict1*10000)

predict2 = np.matmul(np.array([1, 2.5]), theta)
print('For population = 25,000, we predict a profit of ', predict2*10000);


