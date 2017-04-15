#!/usr/bin/env python
# encoding: utf-8

'''
Created on Apr 15, 2017

@author: Yusuke Kawatsu.
'''

import pandas as pd


csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.DataFrame([
    [ 'green', 'M',  10.1, 'class1' ],
    [ 'red',   'L',  13.5, 'class2' ],
    [ 'blue',  'XL', 15.3, 'class1' ]
])

df.columns = [ 'color', 'size', 'price', 'classlabel' ]

print(df, '\n')
