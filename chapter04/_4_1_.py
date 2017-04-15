#!/usr/bin/env python
# encoding: utf-8

'''
Created on Apr 15, 2017

@author: Yusuke Kawatsu.
'''

from io import StringIO

import pandas as pd


csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

print(df, '\n')

print(df.isnull().sum(), '\n')

print(df.values, '\n')
