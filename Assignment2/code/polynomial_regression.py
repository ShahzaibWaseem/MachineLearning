#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt
import math

x_q1 = 0
X_COL_START_IDX = 7

def getData(values, normalize = False):
    global x_q1
    targets = values[:,1]
    x_q1 = values[:,:]

    x = values[:,X_COL_START_IDX:]

    if normalize:
        x = a2.normalize_data(x)

    N_TRAIN = 100;
    x_train = x[0:N_TRAIN,:]
    x_test = x[N_TRAIN:,:]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    # x_train = a2.pd.DataFrame(x_train, index=countries[0:N_TRAIN], columns=features)
    # x_test = a2.pd.DataFrame(x_test, index=countries[N_TRAIN:], columns=features)
    # t_train = a2.pd.Series(t_train)
    # t_test = a2.pd.Series(t_test)

    return x_train, x_test, t_train, t_test

def plotErrors(train_err, test_err, title, log_scale=False):
    # Produce a plot of results.
    if log_scale:
        plt.semilogy(list(train_err.keys()), list(train_err.values()))
        plt.semilogy(list(test_err.keys()), list(test_err.values()))
    else:
        plt.plot(list(train_err.keys()), list(train_err.values()))
        plt.plot(list(test_err.keys()), list(test_err.values()))
    plt.ylabel('RMS')
    plt.legend(['Train error','Test error'])
    plt.title(title)
    plt.xlabel('Polynomial degree')
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
def polynomialFeatures(x, degree):
    features = np.matrix([[1]+[x.item(i, j)**d for d in range(1, degree+1) for j in range(x.shape[1])] for i in range(x.shape[0])])
    return features

# Task 2.2a Polynomial Regression
def polynomialFit(x_train, x_test, t_train, t_test, max_degree = 8):    
    N_TRAIN = x_train.shape[0]
    N_TEST = x_test.shape[0]

    train_err, test_err = {}, {}

    for degree in range(max_degree + 1):
        train_features = polynomialFeatures(x_train, degree)
        w = leastSquares(train_features, t_train)
        train_loss = avgLoss(train_features, t_train, w, N_TRAIN)
        train_err[degree] = train_loss

        test_features = polynomialFeatures(x_test, degree)
        test_loss = avgLoss(test_features, t_test, w, N_TEST)
        test_err[degree] = test_loss

    return train_err, test_err

def main():
    (countries, features, values) = a2.load_unicef_data()
    x_train, x_test, t_train, t_test = getData(values, normalize=False)

    # Task 2.1 Getting Started
    column_idx = np.where(np.asanyarray(features) == "Under-5 mortality rate (U5MR) 1990")[0]
    row_idx = x_q1[:, column_idx].argmax()
    print("Highest Mortality Rate 1990\tCountry:", countries[row_idx], "\t\tMortality:", x_q1[row_idx, column_idx][0, 0])

    column_idx = np.where(np.asanyarray(features) == "Under-5 mortality rate (U5MR) 2011")[0]
    row_idx = x_q1[:, column_idx].argmax()
    print("Highest Mortality Rate 2011\tCountry:", countries[row_idx], "\tMortality:", x_q1[row_idx, column_idx][0, 0])

    # Task 2.2a Polynomial Regression
    train_err, test_err = polynomialFit(x_train, x_test, t_train, t_test)
    plotErrors(train_err, test_err, "Fit with polynomials, no Regularization, No Normalization")
    plotErrors(train_err, test_err, "Fit with polynomials, no Regularization, No Normalization, Log scale RMS", log_scale=True)

    x_train, x_test, t_train, t_test = getData(values, normalize=True)
    train_err, test_err = polynomialFit(x_train, x_test, t_train, t_test)
    plotErrors(train_err, test_err, "Fit with polynomials, no Regularization, Normalization")
    plotErrors(train_err, test_err, "Fit with polynomials, no Regularization, Normalization, Log scale RMS", log_scale=True)

if __name__ == "__main__":
    main()