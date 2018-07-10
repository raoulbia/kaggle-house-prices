#!/usr/bin/env python

import numpy as np
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from matplotlib import pyplot as plt


#### data exploration

def bar_plot_class_var(data):
    sns.countplot(x='Survived',data=data, palette='hls')
    plt.show()

def missing_values(data):
    print(data.isnull().sum())

def distribution_by_feature(data):
    sns.countplot(y="Age", data=data)
    plt.show()

def correlation_features(data):
    # Check the independence between the independent variables
    sns.heatmap(data.corr())
    plt.show()

def scatterplot(data, feature, outcome):
    plt.scatter(x=feature, y=outcome, data=data)
    plt.show()

def regplot(data, feature, outcome):
    sns.regplot(x=feature, y=outcome, data=data)
    plt.show()

def pairplot(data, features, outcome):
    sns.pairplot(data=data,
                 x_vars=features,
                 y_vars=['Survived'])
    plt.show()

def costHistoryPlot(cost_history):
    x = []
    for index, g in enumerate(cost_history):
        x.append(index)
    plt.plot(x, cost_history)
    plt.show()

def plot_learning_curve(errors):

    for i in errors:
        x = []
        for index, g in enumerate(i):
            x.append(index)
        plt.plot(x, i)
    plt.legend(['train', 'test'])
    plt.show()

def plot_validation_curve_reg(reg_param_values, errors):

    # plt.plot(errors[0])
    plt.plot(errors)
    # plt.axis([0, max(reg_param_values), -1, max(errors)]) #[xmin, xmax, ymin, ymax]
    plt.xlabel(reg_param_values)
    plt.grid()
    plt.xlabel("lambda", fontsize=12)
    plt.ylabel("error", fontsize=12)
    plt.title("Validation Curve", fontsize=12)
    plt.legend(['train', 'test'])
    plt.show()

def plot_validation_curve_learn(learn_param_values, errors):
    print(learn_param_values)
    # plt.plot(errors[0])
    plt.plot(learn_param_values, errors)
    # plt.axis([0, max(learn_param_values), -1, max(errors)])  # [xmin, xmax, ymin, ymax]

    # plt.grid()
    # plt.xlabel("alpha", fontsize=12)
    # plt.xlabel(learn_param_values)
    plt.xticks(np.arange(min(learn_param_values), max(learn_param_values)+1)) # range(lower_index, upper_index+1)
    plt.ylabel("error")
    plt.title("Validation Curve")
    plt.legend(['train', 'test'])
    plt.show()