# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import cross_validation,svm
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt
#df=pd.read_csv(r'C:/Users/kartik/Desktop/df_norm_ds.csv')
df=pd.read_csv(r'C:/Users/kartik/Desktop/df_norm_zs.csv')
print(df.head())
df1=df[['Close','SMA','WMA','W%R','S%K','S%D','RSI','CCI','MACD','MACDS']]
#print(df1.head())
X=np.array(df1)
#print(X[0:3])
df1['Label']=df1['Close'].shift(-5)

X_lately=X[-5:]

#print(X_lately[:-1])
X=X[:-5]
#print(X[:2])


y=np.array(df1['Label'][:-5])
#forcast_out=5
#print(forcast_out)
#print(df1.head())


print(len(X),len(y))

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)

pre=clf.predict(X_lately)
print(pre)

'''
plt.figure(figsize=(20,10))
plt.plot(y_test,'b')
plt.plot(pre,'--g')
plt.show()
'''

#print(accuracy)

'''
sm=svm.SVR(kernel='poly')
sm.fit(X_train,y_train)
accuracysvm=sm.score(X_test,y_test)
predict=sm.predict(X_test)
#print(accuracysvm)

plt.figure(figsize=(20,10))
plt.plot(y_test,'r')
plt.plot(predict,'g')
plt.show()

'''















































'''
def min_max_norm(dataframe):
    df=dataframe.copy()
    ti_list=[]
    remove=['Date','Close']

    for x in df.columns:
        if x not in remove:
            ti_list.append(x)
    
    max_range=1
    min_range=-1
    for ti in ti_list:
        max_value = df[ti].max()
        min_value = df[ti].min()
        print(ti)
        df[ti]=df[ti].apply(lambda x:((x-min_value)/(max_value-min_value))*(max_range-min_range)+min_range)
        
    return df
    
#dftemp=min_max_norm(df2)
#print(dftemp.head())


def zscore(dataframe):
    
    df=dataframe.copy()
    ti_list=[]
    remove=['Date','Close']

    for x in df.columns:
        if x not in remove:
            ti_list.append(x)
    
    for ti in ti_list:
        mean = df[ti].mean()
        stdeviation = df[ti].std()
        df[ti]=df[ti].apply(lambda x:(x-mean)/stdeviation)
    
    return df
        
#dftemp=zscore(df2)    
#print(dftemp.head())


def decimal_scaling(dataframe):
    
    df=dataframe.copy()
    ti_list=[]
    remove=['Date','Close']

    for x in df.columns:
        if x not in remove:
            ti_list.append(x)
    
    for ti in ti_list:
        max_value=df[ti].max()
        digit=len(str(int(max_value)))
        df[ti]=df[ti].apply(lambda x:x/(10**digit))
    
    return df

dftemp=decimal_scaling(df2)    
print(dftemp.head())

'''



