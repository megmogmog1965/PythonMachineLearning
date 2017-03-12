#!/usr/bin/env python
# encoding: utf-8

'''
Created on Mar 12, 2017

@author: Yusuke Kawatsu
'''


import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def plot():
    # z samples, -7 to 7.
    z = np.arange(-7, 7, 0.1)
    print(z)
    
    # calc sigmoid values.
    phi_z = sigmoid(z)
    print(phi_z)
    
    # plot.
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.yticks([0.0, 0.5, 1.0])
    
    ax = plt.gca()
    ax.yaxis.grid(True)
    
    plt.show()
    


if __name__ == '__main__':
    plot()
