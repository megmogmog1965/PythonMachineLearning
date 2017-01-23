#!/usr/bin/env python
# encoding: utf-8

'''
Created on Jan 17, 2017

@author: ykawatsu
'''

import numpy as np


class Perceptron(object):
    '''
    classdocs
    '''
    
    def __init__(self, eta=0.01, n_iter=10):
        '''
        Constructor
        '''
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        '''
        :param X: list of (x1, x2, ... )
        :param y: list of y.
        '''
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter): # training loop.
            errors = 0
            
            for xi, target in zip(X, y):
                # update w1, ... wm.
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update * 1
                
                # error count.
                errors += int(update != 0.0)
            
            self.errors_.append(errors)
        
        return self
    
    def net_input(self, X):
        '''
        :return: z
        '''
        return np.dot(X, self.w_[1:]) + 1 * self.w_[0]
    
    def predict(self, X):
        '''
        '''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
