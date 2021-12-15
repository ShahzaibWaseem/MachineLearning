#!/usr/bin/env python

import assignment2 as a2
from polynomial_regression import getData, X_COL_START_IDX
import numpy as np
import matplotlib.pyplot as plt
import math

def plotErrors(train_err, test_err, width=0.25):
    test_keys = np.array([*test_err]) + width
    plt.bar(train_err.keys(), train_err.values(), width = width)
    plt.bar(test_keys, test_err.values(), width = width)
    plt.legend(['Train error','Test error'])
    plt.title("Errors of Regression with respect to one feature")
    plt.xlabel('Feature Number')
    plt.ylabel('RMS')
    plt.show()

# finds the optimal w
# w* = argmin_w L(w) = inv(xTx) xTy = pinv(x)y
def leastSquares(x, y):
    xinv = np.linalg.pinv(x)
    w = xinv * y
    return w

# RMS
# L(w) = (1/N) ||y - xw||^2_2
def avgLoss(x, y, w, N):
    y_hat = x * w
    error = y_hat - y
    loss = math.sqrt((np.transpose(error) * error)/N)
    return loss

# generates a matrix of features (given the degree), by just stacking them horizontally
def polynomialFeatures(x, j, N, degree=3):
    # j is the column number
    features = np.matrix([[1]+[x.item(i, j)**d for d in range(1, degree+1)] for i in range(N)])
    return features

# 8 is the first column, 15 is the last column
def polynomialFit1d(x_train, x_test, t_train, t_test, first_column=8, last_column=15):
    # We have to drop the first 7 columns for x hence subtracting it
    first_column -= X_COL_START_IDX
    last_column -= X_COL_START_IDX

    N_TRAIN = x_train.shape[0]
    N_TEST = x_test.shape[0]

    train_err, test_err = {}, {}

    # -1 becuase we have to account for the 0th column
    for column in range(first_column - 1, last_column):
        train_features = polynomialFeatures(x_train, column, N_TRAIN, degree=3)
        print(train_features.shape)
        w = leastSquares(train_features, t_train)
        train_loss = avgLoss(train_features, t_train, w, N_TRAIN)
        train_err[column+X_COL_START_IDX+1] = train_loss

        test_features = polynomialFeatures(x_test, column, N_TEST, degree=3)
        test_loss = avgLoss(test_features, t_test, w, N_TEST)
        test_err[column+X_COL_START_IDX+1] = test_loss
        
    return train_err, test_err

def main():
    (countries, features, values) = a2.load_unicef_data()
    x_train, x_test, t_train, t_test = getData(values, normalize=False)
    
    train_err, test_err = polynomialFit1d(x_train, x_test, t_train, t_test, first_column=8, last_column=15)
    plotErrors(train_err, test_err, width=0.25)

if __name__ == "__main__":
    main()