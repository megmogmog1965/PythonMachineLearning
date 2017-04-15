#!/usr/bin/env python
# encoding: utf-8

'''
Created on Mar 12, 2017

@author: Yusuke Kawatsu
'''

import numpy as np
import matplotlib.pyplot as plt


def plot():
    np.random.seed(0)
    
    # X.
    X_xor = np.random.randn(200, 2)
    
    # y.
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    
    # plot.
    plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
    
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    
    plt.show()


if __name__ == '__main__':
    plot()
