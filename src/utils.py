# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(name=__name__)

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

from sklearn import model_selection
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from numpy import exp
from scipy.special import expit

def read_csv(train_file_path):
    df = pd.read_csv(train_file_path)
    return df

def split_cleaned_data(df, test_size, dataset):
    # X = np.zeros(shape=(df.shape[0], df.shape[1]))
    # y = np.zeros(shape=(df.shape[0], 1))
    if dataset == 'houses':
        X = np.matrix(df.ix[:, : -1])
        y = np.matrix(df['SalePrice']).T
        X = X.astype(int)
        y = y.astype(int)
        # print(X[:5,:])
    # if dataset == 'houses-toy':
    #     X = np.matrix(df.ix[:, 1:])
    #     y = np.matrix(df['price']).T
    #     X = X.astype(int)
    #     y = y.astype(int)
    #     # print(X[:5,:])

    # add intercept term
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    logger.info('Finished appending intercept term')

    return model_selection.train_test_split(X, y, test_size=test_size)

def sigmoid(p):
    return expit(p)

def evaluate(y, p):
    pass

