#!/usr/bin/env python
# encoding: utf-8

'''
Created on Jan 17, 2017

@author: Yusuke Kawatsu
'''

import numpy as np
from random import seed


class AdalineGD(object):
    '''
    classdocs
    '''
    
    def __init__(self, eta=0.01, n_iter=50):
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
        self.cost_ = []
        
        for i in range(self.n_iter): # training loop.
            
            # calc f=wx.
            output = self.net_input(X)
            
            # difference: y(i) - f(zi).
            errors = (y - output)
            
            # update W1, ... , Wm.
            # dWj = nSIGMA(Yi - f(Zi)) Xij
            self.w_[1:] += self.eta * X.T.dot(errors)
            
            '''
            補足.
            
            [[ X11, X12, X13 ],             [[ Y1 ],       [[ Y1*X11 + Y2*X21 + Y3*X31 ],       [[ dW1 ],
             [ X21, X22, X23 ],   .T.DOT     [ Y2 ],   =    [ Y1*X12 + Y2*X22 + Y3*X32 ]    =    [ dW2 ],
             [ X31, X32, X33 ]]              [ Y3 ]]        [ Y1*X13 + Y2*X23 + Y3*X33 ]]        [ dW3 ]]
            
            '''
            
            # update W0.
            # dW0 = nSIGMA(Yi - f(Zi))
            self.w_[0] += self.eta * errors.sum()
            
            # Cost-function: J(W) = 1/2 SIGMA(Yi - f(Zi))^2
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        
        return self
    
    def net_input(self, X):
        '''
        :return: z
        '''
        return np.dot(X, self.w_[1:]) + 1 * self.w_[0]
    
    def activation(self, X):
        '''
        '''
        return self.net_input(X)
    
    def predict(self, X):
        '''
        '''
        return np.where(self.activation(X) >= 0.0, 1, -1)


class AdalineSGD(object):
    '''
    classdocs
    '''
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        '''
        Constructor
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialilzed = False
        self.shuffle = shuffle
        
        if random_state:
            seed(random_state)
    
    def fit(self, X, y):
        '''
        :param X: list of (x1, x2, ... )
        :param y: list of y.
        '''
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter): # training loop.
            
            if self.shuffle:
                X, y = self._shuffle(X, y)
            
            cost = []
            
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
                
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        
        return self
    
    def partial_fit(self, X, y):
        '''
        '''
        if not self.w_initialilzed:
            self._initialize_weights(X.shape[1])
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        
        else:
            self._update_weights(X, y)
        
        return self
    
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialilzed = True
    
    def _update_weights(self, xi, target):
        '''
        '''
        output = self.net_input(xi)
        error = target - output
        
        # dWj = (Yi - f(Zi))Xij
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        
        # sum-of-squared-error = 1/2 SIGMA(Yi - f(Zi))^2
        cost = 0.5 * error**2 # this is a part of SIGMA (i->n).
        return cost
    
    def net_input(self, X):
        '''
        :return: z
        '''
        return np.dot(X, self.w_[1:]) + 1 * self.w_[0]
    
    def activation(self, X):
        '''
        '''
        return self.net_input(X)
    
    def predict(self, X):
        '''
        '''
        return np.where(self.activation(X) >= 0.0, 1, -1)
