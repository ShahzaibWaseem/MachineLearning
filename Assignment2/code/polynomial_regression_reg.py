#!/usr/bin/env python

import assignment2 as a2
from polynomial_regression import getData, X_COL_START_IDX
import numpy as np
import matplotlib.pyplot as plt
import math

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

def plotErrors(err):
    plt.semilogx(list(err.keys()), list(err.values()))
    plt.legend(['validation error'])
    plt.title('Polynomial regression with Cross Validation (Regularization), Normalized')
    plt.xlabel('Lambda')
    plt.ylabel('RMS')
    plt.show()

# finds the optimal w
# w* = argmin_w L(w) = inv(xTx) xTy = pinv(x)y
def leastSquares(x, y, l):
    I = np.identity(x.shape[1])
    xinv = np.linalg.inv(l * I + x.T * x) * x.T
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
def polynomialFeatures(x, N, degree):
    features = np.matrix([[1]+[x.item(i, j)**d for d in range(1, degree+1) for j in range(x.shape[1])] for i in range(N)])
    return features

def polynomialFitCV(x_train, t_train, degree = 2, N_FOLDS = 10):
    N_TRAIN = x_train.shape[0]
    cv_err = {}

    for l in lambdas:
        cv_fold_err = 0
        for fold in range(N_FOLDS):
            x_train_fold = np.concatenate((x_train[0:fold*N_FOLDS], x_train[(fold+1)*N_FOLDS:]))
            t_train_fold = np.concatenate((t_train[0:fold*N_FOLDS], t_train[(fold+1)*N_FOLDS:]))

            train_features = polynomialFeatures(x_train_fold, N_TRAIN - N_FOLDS, degree=degree)
            w = leastSquares(train_features, t_train_fold, l)

            x_val_fold = x_train[fold*N_FOLDS:(fold+1)*N_FOLDS]
            t_val_fold = t_train[fold*N_FOLDS:(fold+1)*N_FOLDS]

            fold_features = polynomialFeatures(x_val_fold, x_val_fold.shape[0], degree=degree)
            cv_fold_err += avgLoss(fold_features, t_val_fold, w, N_FOLDS)

        # Average validation set error
        cv_err[l] = cv_fold_err / N_FOLDS
    return cv_err

def main():
    (countries, features, values) = a2.load_unicef_data()
    x_train, x_test, t_train, t_test = getData(values, normalize=True)
    
    cv_err = polynomialFitCV(x_train, t_train, degree = 2, N_FOLDS = 10)
    plotErrors(cv_err)   

if __name__ == "__main__":
    main()