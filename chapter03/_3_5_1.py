#!/usr/bin/env python
# encoding: utf-8

'''
Created on Apr 15, 2017

@author: Yusuke Kawatsu
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from chapter03._3_2_1 import _plot_decision_regions


def plot_xor():
    np.random.seed(0)
    
    # X.
    X_xor = np.random.randn(200, 2)
    
    # y.
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    
    # SVM.
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    
    # plot decision regions.
    _plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    
    plt.show()

def plot_iris(gamma):
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
    
    # combined.
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    # SVM.
    svm = SVC(kernel='rbf', random_state=0, gamma=gamma, C=1.0)
    svm.fit(X_train_std, y_train)
    
    # plot.
    _plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.xlabel('petal width [standardized]')
    plt.legend(loc='upper left')
    
    plt.show()


if __name__ == '__main__':
    plot_xor()
    plot_iris(gamma=0.2)
    plot_iris(gamma=100.0)
