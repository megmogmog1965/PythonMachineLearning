#!/usr/bin/env python
# encoding: utf-8

'''
Created on Jan 17, 2017

@author: Yusuke Kawatsu
'''


# built-in modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# my modules.
from perceptron import Perceptron
from adaline import AdalineGD, AdalineSGD


def plot_perceptron():
    ########################### plot 1 ###########################
    
    X, y = _load_iris_data()
    
    # plot 1.
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    
    # plot 2.
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    
    # labels.
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    
    # "hanrei".
    plt.legend(loc='upper left')
    
    plt.show()
    
    ########################### plot 2 ###########################
    
    # trainning.
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    # plot errors.
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    
    # labels.
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    
    plt.show()
    
    ########################### plot 3 ###########################
    
    _plot_decision_regions(X, y, classifier=ppn)
    
    # labels.
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    
    plt.legend(loc='upper left')
    plt.show()

def plot_adaline():
    ########################### load data (X, y) ###########################
    
    X, y = _load_iris_data()
    
    ########################### plot epochs/eta. ###########################
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    
    # eta=0.01
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    
    # plot 1.
    ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    
    # eta=0.0001
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    
    # show.
    plt.show()

def plot_adaline_standardized():
    ########################### standardize ###########################
    
    X_std, y = _load_iris_data_standardized()
    
    ########################### plot 3 ###########################
    
    ada = AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)
    _plot_decision_regions(X_std, y, classifier=ada)
    
    # labels.
    plt.title('Adaline - Gradient Discent')
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    
    plt.legend(loc='upper left')
    plt.show()
    
    plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

def plot_adaline_sgd():
    '''
    '''
    X_std, y = _load_iris_data_standardized()
    
    ada = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
    ada.fit(X_std, y)
    
    # plot.
    _plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # plot2.
    plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()

def _load_iris_data():
    ########################### load data (X, y) ###########################
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    
    # 1-100 rows, #4 col.
    y = df.iloc[0:100, 4].values
    
    # map 1 or -1.
    y = np.where(y == 'Iris-setosa', -1, 1)
    
    # 1-100 rows. #1, #3 cols.
    X = df.iloc[0:100, [0, 2]].values
    
    return X, y

def _load_iris_data_standardized():
    X, y = _load_iris_data()
    ########################### standardize ###########################
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    
    return X_std, y

def _plot_decision_regions(X, y, classifier, resolution=0.02):
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



if __name__ == '__main__':
    plot_perceptron()
    plot_adaline()
    plot_adaline_standardized()
    plot_adaline_sgd()
