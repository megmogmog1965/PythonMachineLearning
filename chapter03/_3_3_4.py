#!/usr/bin/env python
# encoding: utf-8

'''
Created on Mar 12, 2017

@author: Yusuke Kawatsu
'''

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model.logistic import LogisticRegression
import matplotlib.pyplot as plt


def plot():
    # load samples.
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    # split data for testing and training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # standardize.
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # calc.
    weights, params = [], []
    for c in np.arange(-5, 5):
        lr = LogisticRegression(C=10**c, random_state=0)
        lr.fit(X_train_std, y_train)
        
        # classes --> features.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        weights.append(lr.coef_[1])
        params.append(10**c)
    
    # convert.
    weights = np.array(weights)
    
    # plot.
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], label='petal width', linestyle='--')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    
    plt.show()


if __name__ == '__main__':
    plot()
