## ====================  Functions ====================
def featureNormalize(X):
    # FEATURENORMALIZE(X) returns a normalized version of X where
    # the mean value of each feature is 0 and the standard deviation
    # is 1. This is often a good preprocessing step to do when
    # working with learning algorithms.
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = X
    X_norm = X_norm - mu
    X_norm = X_norm / sigma
    return [X_norm, mu, sigma]

def computeCostMulti(X, y, theta):
    # COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y
    m = len(Y)
    hX = (np.matmul(X, theta)).flatten()
    J = (1/(2.0*m)) * np.sum ((hX - Y)**2)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    # taking num_iters gradient steps with learning rate alpha

    #another equation
    #theta = theta - alpha * (1/m) * (((X*theta) - y)' * X)'  
    
    m = len(Y)
    J_history = np.zeros((num_iters, ), dtype=float)
    
    for i in range(num_iters):
        hX = (np.matmul(X, theta)).flatten()
        delta = (1.0/m) * (np.matmul(X.T, (hX - Y))).flatten()
        theta = theta - alpha * delta
        
        # Save the cost J in every iteration  
        J_history[i] = computeCostMulti(X, Y, theta)
        
    return [theta, J_history]

def single_predict(x):
    x_norm = x - mu
    x_norm = x_norm / sigma
    x = np.ones((1, 3), dtype=float)
    x[:, 1:] = x_norm 
    price = ((np.matmul(x, theta)).flatten())[0]
    return price
    
    
## ================ Part 1: Feature Normalization ================
import csv
import numpy as np


print('Loading data ...\n')

## Load Data
# read comma separated data
dataTrack = open('ex1data2.txt')
csvReader = csv.reader(dataTrack)
l = list(csvReader)
m = len(l) # number of training examples
data = np.zeros([m, 3], dtype=float)
data[:] = l

X = data[:, :-1]
Y = data[:, -1]



# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = featureNormalize(X)

temp = X
X = np.ones((m, 3), dtype=float)
X[:, 1:] = temp[:,:]

## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01;
num_iters = 1000;

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, ), dtype=float)
theta, J_history = gradientDescentMulti(X, Y, theta, alpha, num_iters)

# Plot the convergence graph
import matplotlib.pyplot as plt

plt.plot(np.linspace(1, num_iters, num=num_iters), J_history, color = 'green')
plt.title('Convergence Graph')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n', theta)

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

newHouse = np.array([1650, 3])
"""
newHouse_norm = newHouse - mu
newHouse_norm = newHouse_norm / sigma
newHouse = np.ones((1, 3), dtype=float)
newHouse[:, 1:] = newHouse_norm 
"""
price = single_predict(newHouse)


# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n  ', price)

