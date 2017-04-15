#!/usr/bin/env python
# encoding: utf-8

'''
Created on Apr 15, 2017

@author: Yusuke Kawatsu
'''

import numpy as np
import matplotlib.pyplot as plt


def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1 - p))

def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([ p, 1 - p ])

def plot():
    # range=0-->1, step=0.01.
    x = np.arange(0.0, 1.0, 0.01)
    
    # calc entropy, error.
    ent = [ entropy(p) if p != 0 else None for p in x ]
    sc_ent = [ e*0.5 if e else None for e in ent ]
    err = [ error(p) for p in x ]
    
    # draw.
    fig = plt.figure()
    ax = plt.subplot(111)
    
    for i, lab, ls, c, in zip(
        [ ent, sc_ent, gini(x), err ],
        [ 'Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error' ],
        [ '-', '-', '--', '-.' ],
        [ 'black', 'lightgray', 'red', 'green' ]
        ):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([ 0, 1.1 ])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
    
    plt.show()


if __name__ == '__main__':
    plot()
