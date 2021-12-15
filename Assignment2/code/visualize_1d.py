#!/usr/bin/env python

import assignment2 as a2
from polynomial_regression import getData, X_COL_START_IDX
import numpy as np
import matplotlib.pyplot as plt

def plotRegression1d(x_train_col, x_test_col, t_train, t_test, w, feature_name, degree=3):
    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    # x_ev = np.linspace(np.asscalar(min(x_train_col)), np.asscalar(max(x_train_col)), num=500)
    x_ev = np.linspace(min(x_train_col).item(), max(x_train_col).item(), num=500)

    # TO DO:: Put your regression estimate here in place of x_ev.
    # Evaluate regression on the linspace samples.

    y_ev = np.matrix([[x_ev[i] ** d for d in range(0, degree+1)] for i in range(0, len(x_ev))]) * w

    plt.rcParams["figure.figsize"] = (20, 20)
    plt.plot(x_ev, y_ev,'r.-')
    plt.plot(x_train_col, t_train,'bo')
    plt.plot(x_test_col, t_test,'gx')
    plt.title('A visualization of a regression estimate using ' + feature_name)
    plt.legend(['Regression Line','Train points', 'Test points'])
    plt.show()

# finds the optimal w
# w* = argmin_w L(w) = inv(xTx) xTy = pinv(x)y
def leastSquares(x, y):
    xinv = np.linalg.pinv(x)
    w = xinv * y
    return w

# generates a matrix of features (given the degree), by just stacking them horizontally (3 vectors in this case)
def polynomialFeatures(x, j, N, degree=3):
    # j is the column number
    features = np.matrix([[x.item(i, j)**d for d in range(0, degree+1)] for i in range(N)])
    return features

def polynomialFit1d(x_train, x_test, t_train, t_test, features, first_column=11, last_column=13):
    # We have to drop the first 7 columns for x hence subtracting it
    first_column -= X_COL_START_IDX
    last_column -= X_COL_START_IDX

    N_TRAIN = x_train.shape[0]
    N_TEST = x_test.shape[0]

    for column in range(first_column - 1, last_column):
        train_features = polynomialFeatures(x_train, column, N_TRAIN, degree=3)
        w = leastSquares(train_features, t_train)
        plotRegression1d(x_train[:, column], x_test[:, column], t_train, t_test, w, features[column+X_COL_START_IDX], degree=3)

def main():
    (countries, features, values) = a2.load_unicef_data()
    x_train, x_test, t_train, t_test = getData(values, normalize=False)

    polynomialFit1d(x_train, x_test, t_train, t_test, features, first_column=11, last_column=13)

if __name__ == "__main__":
    main()