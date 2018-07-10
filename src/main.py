# -*- coding: utf-8 -*-

import argparse
import logging
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True)

from sklearn import preprocessing

from matplotlib import pyplot as plt

import utils
from linear_regression import *
import utils
import utils_stats
import utils_viz

def model_data(dataset, train_file_path, test_file_path, results_file_path, test_size, num_iters, _alpha, basic,
               learning_curve, validation_curve, _lambda):

    # get data
    df = utils.read_csv(train_file_path)
    print(df.head())

    ### Basic Workflow
    if basic==1:
        # separate features from target
        X, _, y, _ = utils.split_cleaned_data(df=df, test_size=0, dataset=dataset)

        # Normalize features
        X = preprocessing.normalize(X)

        theta_optimized, cost_history = trainLinearRegression(X=X, y=y, learning_rate=_alpha, iterations=num_iters,
                                                              reg_param=_lambda)

        logger.info('coeff: \n{}'.format(theta_optimized.T))

        # Predict on Train data
        p = predictValues(X=X, theta=theta_optimized)
        # print(y[:5])
        # print(p[:5])
        utils.evaluate(y=y, p=p) #TODO complete function

        # Cost History Curve
        utils_viz.costHistoryPlot(cost_history=cost_history)

    ### Advanced Workflow

    # split train / test
    X_train, X_test, y_train, y_test = utils.split_cleaned_data(df=df, test_size=test_size, dataset=dataset)

    # Normalize
    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    # Validation Curves
    if validation_curve == 1:
        # lambda
        validationCurveLambda(X=X_train, y=y_train, X_val=X_test, y_val=y_test, _alpha=_alpha, iterations=num_iters)

        # alpha
        validationCurveAlpha(X=X_train, y=y_train, X_val=X_test, y_val=y_test, iterations=num_iters)


    # Learning Curve
    if learning_curve == 1:
        learningCurve(X=X_train, y=y_train, X_val=X_test, y_val=y_test, _alpha=_alpha, iterations=num_iters,
                      _lambda=_lambda)

    ### Apply to Kaggle test data
    if basic==-1 and learning_curve == -1:
        theta_optimized = np.zeros((X_train.shape[1], 1))
        # get data
        X_test = utils.read_csv(test_file_path)
        # print(X_test.shape)
        # print(X_test.head())

        # get house Ids (for later use)
        X_Ids = X_test['Id']

        # Drop features
        X_test = X_test.drop(['Id'], axis=1)


        # add intercept terms column
        X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)

        # Normalize features
        X_test = preprocessing.normalize(X_test)

        ### Predict on Test data
        p = predictValues(X=X_test, theta=theta_optimized)
        # print(p[:5])

        res = pd.concat((X_Ids, pd.DataFrame(p, dtype=float)), axis=1)
        res.to_csv(results_file_path, index=False, header=['Id', 'SalePrice'])
        logger.info('Done!')



if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str)
    argparser.add_argument('--train-file-path')
    argparser.add_argument('--test-file-path')
    argparser.add_argument('--results-file-path')
    argparser.add_argument('--test-size', type=float)
    argparser.add_argument('--num-iters', type=int)
    argparser.add_argument('--_alpha', type=float)
    argparser.add_argument('--basic', type=int)
    argparser.add_argument('--learning-curve', type=int)
    argparser.add_argument('--validation-curve', type=int)
    argparser.add_argument('--_lambda', type=float)
    args = argparser.parse_args()

    model_data(**vars(args))