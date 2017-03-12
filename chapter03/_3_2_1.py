#!/usr/bin/env python
# encoding: utf-8

'''
Created on Jan 26, 2017

@author: Yusuke Kawatsu
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def perceptron():
    '''
    Chapter 3.2
    '''
    # load samples.
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    print(u'class labels: ', np.unique(y))
    
    # split data for testing and training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # standardize.
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # learn by Perceptron Algorithm.
    ppn = Perceptron(n_iter=40, shuffle=True, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    
    print(u'Misclassified samples: %d' % (y_test != y_pred).sum())
    print(u'Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))
    
    # plot.
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    _plot_decision_regions(X_combined_std, y_combined, classifier=ppn, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.xlabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

def _plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    '''
    '''
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot area.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # make grid points.
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
    
    # predict.
    t = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = classifier.predict(t)
    
    # reshape.
    Z = Z.reshape(xx1.shape)
    
    # plot mt-lines.
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    
    # set ranges.
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    
    # BOLD.
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], s=55, c='', marker='o', alpha=1.0, linewidths=1, label='test set')


if __name__ == '__main__':
    perceptron()
