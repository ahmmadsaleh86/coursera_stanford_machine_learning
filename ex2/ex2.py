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

X = data(:, [1, 2]); #Feature 1 and Feature 2
y = data(:, 3); #Negative or Positive

## ==================== Part 1: Plotting ====================
