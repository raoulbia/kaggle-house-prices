# -*- coding: utf-8 -*-

import argparse
import logging
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(linewidth=1500)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

from sklearn import preprocessing

from matplotlib import pyplot as plt

import utils
from linear_regression import *
import utils
import utils_stats
import utils_viz

def model_data(train_file_path,
               test_file_path,
               results_file_path,
               num_iters,
               learn_hyperparamters,
               _alpha,
               _lambda):

    # get data (cleaned)
    X_df = pd.read_csv(train_file_path)

    X = np.matrix(X_df.ix[:, : -1])
    y = np.matrix(X_df['SalePrice']).T

    # Add INTERCEPT term
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

    # inspect
    logger.info('features\n{}'.format(X[:5, :]))
    logger.info('target\n{}'.format(y[:5]))



    if learn_hyperparamters == 1:
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
        alpha_values = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
        # validationCurve(X=X, y=y, hp_name='alpha', hp_values=alpha_values, iterations=num_iters)


        """ Step 2 Validation Curve lambda

        - use alpha value from previous step
        - find lambda that gives the lowest cross validation error
        """
        lambda_values = [0.000001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
        # validationCurve(X=X, y=y, hp_name='lambda', hp_values=lambda_values, iterations=num_iters, learned_alpha=0.03)

        """ Step 3 Learning Curve
        - use learned values for alpha and lambda
        """
        learningCurve(X=X, y=y, alpha=0.03, _lambda=0.0001, iterations=num_iters)


        # verify that cost decreases / converges
        # _, cost_history = gradientDescent(X=X, y=y, alpha=0.03, _lambda=0.0001, iterations=num_iters)
        # utils_viz.costHistoryPlot(cost_history=cost_history)

    if learn_hyperparamters == -1:
        """Apply to Kaggle Test data"""

        X_test_df = pd.read_csv(test_file_path)
        print(X_test_df.head())

        # Get house Ids (for later use)
        X_Ids = X_test_df['Id']

        # Drop features
        X_test_df = X_test_df.drop(['Id'], axis=1)

        # add intercept terms column
        X_test_df = np.append(np.ones((X_test_df.shape[0], 1)), X_test_df, axis=1)

        # Normalize features
        X_test = preprocessing.normalize(X_test_df)
        print(X_test[:5, :3])

        # Get theta
        theta, _ = gradientDescent(X=X,
                                   y=y,
                                   alpha=_alpha,
                                   _lambda=_lambda,
                                   iterations=num_iters)

        ### Predict on Test data
        p = predictValues(X=X_test, theta=theta)
        print(p[:5])

        res = pd.concat((X_Ids, pd.DataFrame(p, dtype=float)), axis=1)
        res.to_csv(results_file_path, index=False, header=['Id', 'SalePrice'])
        logger.info('Done!')



if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train-file-path')
    argparser.add_argument('--test-file-path')
    argparser.add_argument('--results-file-path')
    argparser.add_argument('--num-iters', type=int)
    argparser.add_argument('--learn-hyperparamters', type=int)
    argparser.add_argument('--_alpha', type=float)
    argparser.add_argument('--_lambda', type=float)



    args = argparser.parse_args()

    model_data(**vars(args))