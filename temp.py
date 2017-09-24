# -*- coding: utf-8 -*-



import numpy as np 
a=np.array([1,2,3,4])
#print(a)

a=np.delete(a,a[0:2])
#print(a)


from pandas_datareader import data
from datetime import datetime
import pandas as pd

#ibm = data.DataReader('IBM',  'yahoo', datetime(2000, 1, 1))

#moving average
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
'''
a=movingaverage(ibm['Close'],10)
a=np.delete(a,np.s_[:5])

a=np.delete(a,range(a.size-5,a.size))

b=np.zeros(10)
b.fill(np.nan)
a=np.concatenate([b,a])

print(a[0:20])
ibm['SMA'] = pd.Series(a, index=ibm.index)

print(ibm.head(15))

'''
print((data.DataReader("AAPL",'yahoo',datetime(2000,1,1))).head())
