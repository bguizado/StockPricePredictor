# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd

def movingaverage(close, window_size):
    window = np.ones(int(window_size))/float(window_size)
    ma= np.convolve(close, window, 'same')
    ma=np.delete(ma,np.s_[:int((window_size)/2)])
    ma=np.delete(ma,range(ma.size-int((window_size)/2),ma.size))
    b=np.zeros(close.size-ma.size)
    b.fill(np.nan)
    ma=np.concatenate([b,ma])
    
    return ma


def weightedmovingaverage(close,window_length):
    i=0
    wm=[]
    weight=np.arange(1,window_length+1)
    #print(weight)
    
    
    while(i+len(weight)<len(close)):
        b=sum(np.multiply(close[i:i+len(weight)],weight))
        wm.append(b/sum(weight))
        #print(wma)
        i=i+1
        
        
    wma=np.array(wm)    
    x=np.zeros(close.size-wma.size)
    x.fill(np.nan)
    
    wma=np.concatenate([x,wma])
     
    return wma


    
def ema(close, window_length):

    ema = []
    j = 1
    sma = sum(close[:window_length]) / window_length
    multiplier = 2 / float(1 + window_length)
    ema.append(sma)

    #EMA(current) = ( (Close - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (close[window_length] - sma) * multiplier) + sma)
    
    for i in close[window_length+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema


    
def relativestrengthindex(close,window_length):
    delta=close.diff()
    delta = delta[1:] 
    delta = delta[1:] 
    gain,loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0

    #print("gain")
    #print(gain[0:10])
    
    loss[loss > 0] = 0
    #print("loss")
    #print(loss[0:10])
    
    sma_gain = gain.rolling(window=window_length,center=False).mean()
    #print("sma_gain")
    #print(sma_gain[10:15])
    
    
    sma_loss = (loss.abs()).rolling(window=window_length,center=False).mean()
    #print("sma_loss")
    #print(sma_loss[10:15])
    
    RS = sma_gain/sma_loss
    #print("RS")
    #print(RS[10:15])
    
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    return RSI


    
def stochastic_k(close,window_length):
    low = close.rolling(window=window_length,center=False).min()
    #print(low.size)
    high= close.rolling(window=window_length,center=False).max()
    #print(high.size)
    k = 100 * (close - low) / (high - low)
    return k
    

    
def stochastic_d(sto_k,window_length):
    return sto_k.rolling(window=window_length,center=False).mean()
  
    

    
def willams_r(high,low,close,window_length):
    highest_high = high.rolling(window=window_length,center=False).max()
    #print(highest_high[15])
    lowest_low= low.rolling(window=window_length,center=False).min()
    #print(lowest_low[15])
    
    wr = (-100) * (highest_high-close) / (highest_high - lowest_low)
    return wr


    
def commoditychannelindex(high,low,close,window_length):
    
    listoflist=[high,low,close]
    tp=[sum(x) for x in zip(*listoflist)]
    tp=np.divide(tp,3)
    
    const=0.015
    window = np.ones(int(window_length))/float(window_length)
    sma_tp=np.convolve(tp, window, 'same')
    #sma_tp = tp.rolling(window=window_length,center=False).mean()
    sma_tp=np.delete(sma_tp,np.s_[:10])
    sma_tp=np.delete(sma_tp,range(sma_tp.size-10,sma_tp.size))
    b=np.zeros(window_length)
    b.fill(np.nan)
    
    sma_tp=np.concatenate([b,sma_tp])
    
    sub=np.subtract(tp,sma_tp)
    '''
    print(tp[18:23])
    print(sma_tp[18:23])
    print(sub[18:23])
    '''
    sub_abs=np.abs(sub)
    #print("abs")
    sub_abs_sum=sum(sub_abs[20:])
    sd=sub_abs_sum/window_length
    
    return np.divide(np.subtract(tp,sma_tp),(const*sd))
    

    
def macd(close):
    
    ema12=np.array(ema(close,12))
    ema26=np.array(ema(close,26))
    
    #print(ema12.size)
    #print(ema26.size)
    
    z12=np.zeros(close.size-ema12.size)
    z12.fill(np.nan)
    
    z26=np.zeros(close.size-ema26.size)
    z26.fill(np.nan)
    
    ema12=np.concatenate([z12,ema12])
    ema26=np.concatenate([z26,ema26])
    
    '''
    print(ema12[27])
    print(ema26[27])
    '''
    md=np.subtract(ema12,ema26)
    #print(md[z26.size-1])
    mdsignal=np.array(ema(md[z26.size:],9))
    zmds=np.zeros(close.size-mdsignal.size)
    zmds.fill(np.nan)
    
    mdsignal=np.concatenate([zmds,mdsignal])
    #print(zmds.size)
    return md,mdsignal

