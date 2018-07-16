# -*- coding: utf-8 -*-

import argparse
import logging

logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(linewidth=1200)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

from sklearn import preprocessing
from sklearn.datasets import load_boston

from linear_regression import *
import utils_viz

def model_data(num_iters, alpha, _lambda):

    # get data
    boston = load_boston()
    # print(boston.DESCR)

    # split features from target
    X = boston.data
    y = boston.target

    # Normalize features
    X = preprocessing.normalize(X)
    logger.info('finished normalizing input data')

    # make y a [m x 1] col vector
    y = y.reshape((len(y), 1))


    # inspect
    logger.info('features\n{}'.format(X[:5, :]))
    logger.info('target\n{}'.format(y[:5]))

    """
    Manual hyperparamter learning

    NOTE: THIS IS USEFUL FOR LEARNING HOW THINGS WORK BUT IN PRACTICE FINDING BEST HYPERPARAMS MANUALLY IS VERY TRICKY
          PROBABLY BEST TO RESORT TO GRID SEARCH FOR THIS TASK

    Step 1: find a good alpha - recall that lambda is for generalization so to find alpha we don't need lambda
    Step 2: find a good lambda using learned alpha
    Step 3: inspect learning curve using learned hyperparamters
    """

    """ Step 1 Validation Curve alpha

    - If we record the learning at each iteration and plot the learning rate (log) against loss;
    - we will see that as the learning rate increase, there will be a point where the loss stops decreasing
    and starts to increase.
    - in practice, our learning rate should ideally be somewhere to the left to the lowest point of the graph
    source: https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
    """
    alpha_values = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10, 20]
    # validationCurve(X=X, y=y, hp_name='alpha', hp_values=alpha_values, iterations=num_iters)


    """ Step 2 Validation Curve lambda

    - use alpha value from previous step
    - find lambda that gives the lowest cross validation error
    """
    lambda_values = [0.000001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
    # validationCurve(X=X, y=y, hp_name='lambda', hp_values=lambda_values, iterations=num_iters, learned_alpha=0.01)

    """ Step 3 Learning Curve
    - use learned values for alpha and lambda
    """
    learningCurve(X=X, y=y, alpha=0.01, _lambda=0.1, iterations=num_iters)

    # verify that cost decreases / converges
    _, cost_history = gradientDescent(X=X,
                                      y=y,
                                      alpha=alpha,
                                      _lambda=_lambda,
                                      iterations=num_iters)

    # utils_viz.costHistoryPlot(cost_history=cost_history)





if __name__ == '__main__':

    model_data(num_iters=1500, alpha=0.01, _lambda=0.01)