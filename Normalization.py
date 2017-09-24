# -*- coding: utf-8 -*-


import pandas as pd

def min_max_norm(dataframe):
    df=dataframe.copy()
    
    
    max_range=1
    min_range=-1
    for ti in df.columns:
        max_value = df[ti].max()
        min_value = df[ti].min()
        print(ti)
        df[ti]=df[ti].apply(lambda x:((x-min_value)/(max_value-min_value))*(max_range-min_range)+min_range)
        
    return df
    
def zscore(dataframe):
    
    df=dataframe.copy()
    
    for ti in df.columns:
        mean = df[ti].mean()
        stdeviation = df[ti].std()
        df[ti]=df[ti].apply(lambda x:(x-mean)/stdeviation)
    
    return df
    
    
def decimal_scaling(dataframe):
    
    df=dataframe.copy()
    
    '''
    ti_list=[]
    remove=['Date','Close']

    for x in df.columns:
        if x not in remove:
            ti_list.append(x)
    '''
    for ti in df.columns:
        max_value=df[ti].max()
        digit=len(str(int(max_value)))
        df[ti]=df[ti].apply(lambda x:x/(10**digit))
    
    return df