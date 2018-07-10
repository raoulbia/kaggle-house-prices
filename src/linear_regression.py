# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(precision=13)
np.set_printoptions(suppress=True)
import utils_viz


def validationCurveAlpha(X, y, X_val, y_val, iterations):

    learn_param_values = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

    error_train = [] #np.zeros((m, 1))
    error_test = [] #np.zeros((m, 1))

    for learn_param in learn_param_values:

        print(X[:5,:])
        # learn theta_optimized
        theta_optimized, _ = trainLinearRegression(X=X, y=y,
                                                   learning_rate=learn_param,
                                                   iterations=iterations,
                                                   reg_param=0)

        cost_te, _ = regularizedCostFunction(X=X_val, theta=theta_optimized, y=y_val,
                                             learning_rate=1,
                                             reg_param=0)

        error_test.append(np.asscalar(cost_te))

    print(error_test)
    utils_viz.plot_validation_curve_alpha(learn_param_values, errors=error_test)


def validationCurveLambda(X, y, X_val, y_val, _alpha, iterations):

    reg_param_values = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_test = [] #np.zeros((m, 1))

    for reg_param in reg_param_values:

        # learn theta_optimized
        theta_optimized, _ = trainLinearRegression(X=X, y=y,
                                                   learning_rate=_alpha,
                                                   iterations=iterations,
                                                   reg_param=reg_param)

        cost_te, _ = regularizedCostFunction(X=X_val, theta=theta_optimized, y=y_val,
                                             learning_rate=1,
                                             reg_param=0)

        error_test.append(np.asscalar(cost_te))

    utils_viz.plot_validation_curve_lambda(reg_param_values, errors=error_test)



def learningCurve(X, y, X_val, y_val, _alpha, iterations, _lambda):
    m = X.shape[0]
    # m = 30
    error_train = [] #np.zeros((m, 1))
    error_test = [] #np.zeros((m, 1))

    for subset_size in range(1, m, 100):
        X_train = X[:subset_size, :]
        y_train = y[: subset_size]

        # learn theta_optimized
        theta_optimized, _ = trainLinearRegression(X=X_train, y=y_train,
                                                   learning_rate=_alpha,
                                                   iterations=iterations,
                                                   reg_param=_lambda)

        cost_tr, _ = regularizedCostFunction(X=X_train, theta=theta_optimized, y=y_train,
                                             learning_rate=1,
                                             reg_param=0)

        cost_te, _ = regularizedCostFunction(X=X_val, theta=theta_optimized, y=y_val,
                                             learning_rate=1,
                                             reg_param=0)

        error_train.append(np.asscalar(cost_tr))
        error_test.append(np.asscalar(cost_te))



    # print(error_train)
    # print(error_test)
    utils_viz.plot_learning_curve(errors=[error_train, error_test])


def trainLinearRegression(X, y, learning_rate, iterations, reg_param):
    cost_history = []
    theta = np.zeros((X.shape[1], 1)) # init theta col. vector
    for i in range(iterations):
        J, gradient = regularizedCostFunction(X, theta, y, learning_rate, reg_param)
        theta = theta - gradient
        cost_history.append(J)
    return theta, cost_history


def regularizedCostFunction(X, theta, y, learning_rate, reg_param):

    # init useful vars
    m = y.shape[0] # number of training examples

    predictions = X @ theta
    # print(y[:5].T)
    # print(predictions[:5].T, '\n')

    delta = predictions - y # [m x 1]
    gradient = (learning_rate / m) * X.T * delta  # [n x 1] = [n x m] x [m x 1]
    regularization = (reg_param  / m) * theta  # [n x 1]
    regularization[0] = 0 # don't regularize intercept term
    # logger.info('regularization: {}'.format(regularization.T))
    gradient = gradient + regularization

    # compute cost
    squared_errors = np.sum(np.square(delta))
    regularization = np.sum(reg_param * np.square(theta[1:]))
    J = (squared_errors + regularization) / (2*m)
    return J, gradient


def predictValues(X, theta):
    return X @ theta  # [m x 1] = [m x n] x [n x 1]

