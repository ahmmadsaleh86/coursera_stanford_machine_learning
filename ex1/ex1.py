## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     warmUpExercise
#     plotData.m
#     gradientDescent.m
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

def plotData(X, Y):
    #visualize
    plt.scatter(X[:,0], Y, color = 'red')
    plt.title('Population VS Profite')
    plt.xlabel('Populaion in $10,000s')
    plt.ylabel('profite in $10,000s')
    plt.show()

"""    
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate alpha

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    k = 1:m;
    t1 = sum((theta(1) + theta(2) .* X(k,2)) - y(k)); % Un-Vectorized
    t2 = sum(((theta(1) + theta(2) .* X(k,2)) - y(k)) .* X(k,2)); % Un-Vectorized
    
    theta(1) = theta(1) - (alpha/m) * (t1);
    theta(2) = theta(2) - (alpha/m) * (t2);
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
"""

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
plotData(data[:, :-1], Y);

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')
theta = np.zeros((2, 1), dtype=float); # initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

"""
# compute and display initial cost
computeCost(X, Y, theta)

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 1.8] *theta;
fprintf('For population = 18,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 2.5] * theta;
fprintf('For population = 25,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;
"""

