# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
df=pd.read_csv(r'C:/Users/kartik/Desktop/temp.csv')
close = df['Close']
high=df['High']
low=df['Low']
#print("close size")
print(close[13])

def ema(close, window_length):

    ema = []
    j = 1
    sma = sum(close[:window_length]) / window_length
    multiplier = 2 / float(1 + window_length)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (close[window_length] - sma) * multiplier) + sma)
    
    for i in close[window_length+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema
    

def stochastic_k(close,window_length):
    low = close.rolling(window=window_length,center=False).min()
    print(low.size)
    high= close.rolling(window=window_length,center=False).max()
    print(high.size)
    k = 100 * (close - low) / (high - low)
    return k
    

def stochastic_d(sto_k,window_length):
    return sto_k.rolling(window=window_length,center=False).mean()
    
    
    
def willams_r(high,low,close,window_length):
    highest_high = high.rolling(window=window_length,center=False).max()
    print(highest_high[15])
    lowest_low= low.rolling(window=window_length,center=False).min()
    print(lowest_low[15])
    
    wr = (-100) * (highest_high-close) / (highest_high - lowest_low)
    return wr
    
def cci(high,low,close,window_length):
    
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
    print(tp[18:23])
    print(sma_tp[18:23])
    print(sub[18:23])
    
    sub_abs=np.abs(sub)
    #print("abs")
    sub_abs_sum=sum(sub_abs[20:])
    sd=sub_abs_sum/window_length
    
    return np.divide(np.subtract(tp,sma_tp),(const*sd))
 

def rsi(close,window_length):
    delta=close.diff()
    delta = delta[1:] 
    delta = delta[1:] 
    gain,loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0

    print("gain")
    print(gain[0:10])
    
    loss[loss > 0] = 0
    print("loss")
    print(loss[0:10])
    
    sma_gain = gain.rolling(window=window_length,center=False).mean()
    print("sma_gain")
    print(sma_gain[10:15])
    
    
    sma_loss = (loss.abs()).rolling(window=window_length,center=False).mean()
    print("sma_loss")
    print(sma_loss[10:15])
    
    RS = sma_gain/sma_loss
    print("RS")
    print(RS[10:15])
    
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    return RSI

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
    
    print(ema12[27])
    print(ema26[27])
    
    md=np.subtract(ema12,ema26)
    print(md[z26.size-1])
    mdsignal=np.array(ema(md[z26.size:],9))
    zmds=np.zeros(close.size-mdsignal.size)
    zmds.fill(np.nan)
    
    zmds=np.concatenate([zmds,mdsignal])
    print(zmds.size)
    return md,zmds

    
    
def wma(close,window_length):
    i=0
    wm=[]
    weight=np.arange(1,window_length+1)
    print(weight)
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
    

    

    
    
    
    

#temp=stochastic_k(close,14)
#print(temp.size)

#temp1=stochastic_d(temp,3)
#print(temp1.size)
    
#temp2= willams_r(high,low,close,14)
#print(temp2[15])
#print(close[15])

#temp3=cci(high,low,close,20)
#print(temp3[0:100])


#temp4=macd(close)
#print(temp4.size)
#print(temp4[27])

temp5=np.array(wma(close,10))
print(temp5[9:11])
print(temp5.size)

'''
n=close[0]*1+close[1]*2+close[2]*3+close[3]*4+close[4]*5+close[5]*6+close[6]*7+close[7]*8+close[8]*9+close[9]*10
print(n)
print(sum(np.arange(1,11)))
print(n/sum(np.arange(1,11)))

'''   


