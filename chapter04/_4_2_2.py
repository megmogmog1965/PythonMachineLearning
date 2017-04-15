#!/usr/bin/env python
# encoding: utf-8

'''
Created on Apr 15, 2017

@author: Yusuke Kawatsu.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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

inv_class_mapping = { label: idx for idx, label in enumerate(np.unique(df['classlabel'])) }
print(inv_class_mapping, '\n')

# df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df, '\n')

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y, '\n')

print(class_le.inverse_transform(y), '\n')
