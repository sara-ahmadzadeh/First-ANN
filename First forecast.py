# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:10:53 2016

@author: Sara
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pandas import Series, DataFrame, Panel
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test : '
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput


def make_stationarity(timeseries):
    #log transformation
    ts_log = np.log(timeseries)    
    #decopmosation
    decomposition = seasonal_decompose(ts_log)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    return ts_log_decompose


def differencing(timeseries):
    ts_log_diff = timeseries - timeseries.shift()
    return ts_log_diff


def plot_ACF_PACF(timesries):
    lag_acf = acf(timesries, nlags=20)
    lag_pacf = pacf(timesries, nlags=20, method='ols')
    #Plot ACF
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timesries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timesries)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timesries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timesries)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout() 


def ARIMA_model(timeseries,p,d,q):
    model = ARIMA(timeseries, order=(p, d, q))  
    results_ARIMA = model.fit(disp=-1)  
    return results_ARIMA
    

def back_to_original(arimapredict,timeserieslog):
    predictions_ARIMA_diff = pd.Series(arimapredict.fittedvalues, copy=True)
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_log = pd.Series(timeserieslog.ix[0], index=timeserieslog.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    return predictions_ARIMA


if __name__ == '__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse)
    ts = data['#Passengers']
#    test_stationarity(ts)
#==============================================================================
#     stationary_ts = make_stationarity(ts)
#     test_stationarity(stationary_ts)
#     plot_ACF_PACF(stationary_ts)
#     ARIMA = ARIMA_model(stationary_ts,2,0,2)
#==============================================================================
    ts_log = np.log(ts)
    ts_log_diff = differencing(ts_log)
    ARIMA_predict = ARIMA_model(ts_log,2,1,2)
#==============================================================================
#     plt.plot(ts_log_diff, color='blue', label='target')
#     plt.plot(ARIMA_predict.fittedvalues, color='red', label='ARIMA_predict')
#     plt.legend(loc='best')
#     plt.title('RSS: %.4f'% sum((ARIMA_predict.fittedvalues-ts_log_diff)**2))
#==============================================================================
    final_prediction_ARIMA = back_to_original(ARIMA_predict,ts_log)
    plt.plot(ts, color='blue', label='target')
    plt.plot(final_prediction_ARIMA, color='red', label='predict')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f'% np.sqrt(sum((final_prediction_ARIMA-ts)**2)/len(ts)))
    
        
        
    
    
