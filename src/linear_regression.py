#!/usr/bin/env python

import logging
logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(precision=13)
np.set_printoptions(suppress=True)

import utils


def learningCurveLinReg(X, y, X_val, y_val, learning_rate, iterations, reg_param):
    m = X.shape[0]
    # m = 30
    error_train = [] #np.zeros((m, 1))
    error_test = [] #np.zeros((m, 1))

    # i = 0
    for subset_size in range(1, m, 100):
        X_train = X[:subset_size, :]
        y_train = y[: subset_size]

        # learn theta_optimized
        theta_optimized, _ = trainLinReg(X=X_train, y=y_train,
                               learning_rate=learning_rate,
                               iterations=iterations,
                               reg_param=reg_param)

        cost_tr, _ = regularizedCostLinReg(X=X_train, theta=theta_optimized, y=y_train,
                                           learning_rate=1,
                                           reg_param=0)

        cost_te, _ = regularizedCostLinReg(X=X_val, theta=theta_optimized, y=y_val,
                                           learning_rate=1,
                                           reg_param=0)

        error_train.append(np.asscalar(cost_tr))
        error_test.append(np.asscalar(cost_te))
        # i += 1

    return error_train, error_test


def trainLinReg(X, y, learning_rate, iterations, reg_param):
    cost_history = []
    theta = np.zeros((X.shape[1], 1)) # init theta col. vector
    for i in range(iterations):
        J, gradient = regularizedCostLinReg(X, theta, y, learning_rate, reg_param)
        theta = theta - gradient
        cost_history.append(J[0,0])
    return theta, cost_history


def regularizedCostLinReg(X, theta, y, learning_rate, reg_param):

    # init useful vars
    m = y.shape[0] # number of training examples

    predictions = X @ theta
    # print(y[:5].T)
    # print(predictions[:5].T, '\n')

    delta = predictions - y # delta is an [m x 1] column vector

    theta_rest = theta[1:] # [(n-1) x 1]

    grad_intercept = learning_rate / m * X[:, 0 ].T * delta  # don't regularize intercept term
    grad_rest      = learning_rate / m * X[:, 1:].T * delta  # [(n-1) x 1] = [(n-1) x m] x [m x 1] + [(n-1) x 1]
    reg_term       = reg_param * 1/m * theta_rest # [(n-1) x m]
    grad_rest      = grad_rest + reg_term
    gradient = np.concatenate((grad_intercept, grad_rest))

    # compute cost
    J = 1 / (2*m) * sum(np.square(delta))
    # or
    # J = 1 / (2 * m) * delta.T * delta

    reg_term = reg_param /(2*m) * sum(np.square(theta_rest))
    J = J + reg_term
    # print(J)
    return J, gradient


def predictLinReg(X, theta):

    return X * theta  # [m x 1] = [m x n] x [n x 1]

