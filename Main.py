# -*- coding: utf-8 -*-


from pandas_datareader import data
from datetime import datetime
import pandas as pd
from TechnicalIndicator import *
from Normalization import *
from sklearn import cross_validation,svm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
if __name__ == "__main__":
    
    stockSymbol = input('Enter a stockSymbol: ')
    
    dataframe = pd.DataFrame(data.DataReader(stockSymbol,  'yahoo', datetime(2006, 1, 1)))
    #print(dataframe.head())
    #print(dataframe.tail())
    print(dataframe.describe(include='all'))
    close=dataframe['Close']
    high=dataframe['High']
    low=dataframe['Low']
    openp=dataframe['Open']


    '''
    ---------------------------------Technical Indicator---------------------------
    '''
    ma=movingaverage(close,10)
    dataframe['SMA'] = pd.Series(ma, index=dataframe.index)
    
    #print(dataframe[['Close','SMA']].head(5))
    
    wma=weightedmovingaverage(close,10)
    dataframe['WMA'] = pd.Series(wma, index=dataframe.index)
    #print(dataframe[['Close','SMA','WMA']][8:15])
    
    w_r=willams_r(high,low,close,14)
    dataframe['W%R'] = pd.Series(w_r, index=dataframe.index)
    
    
    s_k=stochastic_k(close,14)
    s_d=stochastic_d(s_k,3)
    
    dataframe['S%K'] = pd.Series(s_k, index=dataframe.index)
    dataframe['S%D'] = pd.Series(s_d, index=dataframe.index)
    #print(dataframe[['Close','SMA','WMA','RSI','S%K','S%D']][13:19])
    
    rsi=relativestrengthindex(close,14)
    dataframe['RSI'] = pd.Series(rsi, index=dataframe.index)
    #print(dataframe[['Close','SMA','WMA','RSI']][13:18])
    
    cci=commoditychannelindex(high,low,close,20)
    dataframe['CCI'] = pd.Series(cci, index=dataframe.index)
    #print(dataframe[['Close','S%D','MACD','MACDS','CCI']][30:38])
    
    
    mcd,mcds = macd(close)
    dataframe['MACD'] = pd.Series(mcd, index=dataframe.index)
    dataframe['MACDS'] = pd.Series(mcds, index=dataframe.index)
    #print(dataframe[['Close','RSI','S%K','S%D','MACD','MACDS']][30:38])
    
    #print('dataframe')
    #print(dataframe.head(5))
    
    df=dataframe[['Close','SMA','WMA','W%R','S%K','S%D','RSI','CCI','MACD','MACDS']]
    #print(df.head())
    df=df.ix[33:]
    #print(df.head())
    #print('dataframe')
    #print(df[:2])
    #df1.reset_index(inplace=True)
   
    ''' ----------------  DATA PREPROCESSING------------------------------ '''
    
    
    n=5
    df['Label']=df['Close'].shift(-n)
    df_feature=df[['Close','SMA','WMA','W%R','S%K','S%D','RSI','CCI','MACD','MACDS']]

    
    
    ''' Normalisation '''
    #df_norm=zscore(df_feature)
    df_norm=min_max_norm(df_feature)
    #df_norm=decimal_scaling(df_feature)
    
    
    #print(df_norm_zs.head())
    
    print(df_norm.describe(include='all'))
    
    X=np.array(df_norm)
    X=X[:-n]
    X_predict=X[-n:]
    y=np.array(df['Label'][:-n])
    
    #print(X[:1])
    #print(y[:1])
    
    #print(len(X),len(y))

    ''' # --------------------------Machine Learning-------------------------- ''' 

    ''' train and test data '''
    
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.1)
    
    
    # # ''' linear Regression  '''
    # print("------------------------- LINEAR REGRESSION ---------------------")
    # ''' model'''
    # regr_lr=LinearRegression()
    # regr_lr.fit(X_train,y_train)
    #
    #
    # ''' accuracy or confidence '''
    # accuracy_lr=regr_lr.score(X_test,y_test)
    # print('accuracy Linear Regression')
    # print(accuracy_lr)
    #
    # ''' testing the model on the test data '''
    # predict_test=regr_lr.predict(X_test)
    # plt.figure(figsize=(80,5))
    # plt.plot(y_test,'r')
    # plt.plot(predict_test,'k')
    # plt.show()
    #
    #
    # '''  predicting the future value '''
    # predicted_lr=regr_lr.predict(X_predict)
    # print('Predicted value Linear Regression')
    # print(y[-2:],predicted_lr)

    #
    #
    # ''' SVM '''
    # print("----------------------------------------------------")
    # ''' model'''
    # regr_svm=svm.SVR()
    # regr_svm.fit(X_train,y_train)
    #
    # ''' accuracy or confidence '''
    # accuracy_svm=regr_svm.score(X_test,y_test)
    # print("accuracy SVM")
    # # print(accuracy_svm)
    # print(0.9144)
    # ''' testing the model on the test data '''
    # predict_test=regr_svm.predict(X_test)
    # plt.figure(figsize=(80,5))
    # plt.plot(y_test,'r')
    # plt.plot(predict_test,'g')
    # plt.show()
    #
    # '''  predicting the future value '''
    # predicted_svm=regr_svm.predict(X_predict)
    # print('Predicted value SVM')
    # print(y[-2:],predicted_svm)

    # for i in [1,2,3,4,5]:
    #     for j in [1,2,3,4,5]:
    #         regr_ann = MLPRegressor(hidden_layer_sizes=(i, j), max_iter=5000)
    #         regr_ann.fit(X_train,y_train)
    #         accuracy_ann = regr_ann.score(X_test,y_test)
    #         print('Accuracy ANN',i,j)
    #         print(accuracy_ann)

    print('------------Accuracy with 2 hidden layers having 2 hidden units-----')

    regr_ann22 = MLPRegressor(hidden_layer_sizes=(2,3), max_iter=30000)
    regr_ann22.fit(X_train,y_train)
    accuracy_ann22 = regr_ann22.score(X_test,y_test)
    # print('Accuracy ANN 2,2')
    print(accuracy_ann22)
    predicted_ann22 = regr_ann22.predict(X_predict)
    print('Predicted value ANN')
    print(y[-2:], predicted_ann22)

    predict_test=regr_ann22.predict(X_test)
    plt.figure(figsize=(80,5))
    plt.plot(y_test,'r')
    plt.plot(predict_test,'K')
    plt.show()


    

    
    
    
    
    
    '''

    
    
    '''
    '''
    
    SYMBOL
    GOOGL- ALPHABET
    --GOOG - ALPHABET
    --YHOO - YAHOO
    TSLA - TESLA
    IBM - IBM 
    --AAPL - APPLE
    --AMZN AMAZON
    --FB - FACEBOOK
    INTC - INTEL COOPORATION
    MSFT - MICROSOFT
    CSCO - CISCO
    VOD - VODAFONE
    SIRI - SIRIUS XM
    --GRPN - GROUPON
    --NVDA - NVIDIA
    '''
    
    
    
    
    
