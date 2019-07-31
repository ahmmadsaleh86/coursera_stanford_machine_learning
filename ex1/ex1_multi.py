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
print('Normalizing Features ...\n');

X, mu, sigma = featureNormalize(X)

temp = X
X = np.ones((m, 3), dtype=float)
X[:, 1:] = temp[:,:]

