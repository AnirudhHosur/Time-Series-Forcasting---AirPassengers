# Time Series Forcasting
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# read the data
data = pd.read_csv('AirPassengers.csv')
print(data.head())
print(data.dtypes)

# Read the data is series format
# series = pd.read_csv('AirPassengers.csv', header=0, index_col=0, squeeze=True)

# Use the month column as our index
from datetime import datetime
data['Month']=pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
#check datatype of index
data.index

#convert to time series:
ts = data['#Passengers']
ts.head(10)

# Checking the plot
plt.plot(ts)

# To apply time series to a model, all its statistical properties should be 
# constant over time -> mean, variance

# QUick and dirty way to make sure the data is stationary is by dividing the 
# data and checking the mean and variance of each group
ts.hist()

X = ts.values
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# Hence, non-stationary time series
# One way is to log transform -> to flatten out the exponential growth
from numpy import log
X = np.log(X)
plt.plot(X)

# Calculate the mean and variance of the log transformed data
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# The Augumented Dicky-Fuller test
# Null Hypothisis -> (accept) -> p>0.05 -> Non-stationary
#                -> (reject) -> p<=0.05 -> statioanry

from statsmodels.tsa.stattools import adfuller
def test_adFuller(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))
test_adFuller(ts)
    
# Try again on logged data
X = np.log(X)
test_adFuller(X)


# Say with certainity that data is non-stationary

# NOW TIME TO REMOVE THESE TREND AND SEASONALITIES
# Making the non-stationary time series stationary -->

# 1) Taking care of trends

# Log again
ts_log = np.log(ts)
plt.plot(ts_log)

# Smoothing the trend here using 2 methods ->
# Moving average and exponentially weighted moving average

# Moving average
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

# Subtract rolling mean from original series
ts_log_rolling_mean_diff = ts_log - moving_avg
ts_log_rolling_mean_diff.dropna(inplace=True)

test_adFuller(ts_log_rolling_mean_diff)

# p-value is lower

# Exponentially weighted moving averages
exp_wt_avg = ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(exp_wt_avg, color='blue')

# AGain subtract
ts_log_exp_diff = ts_log - exp_wt_avg
test_adFuller(ts_log_exp_diff)

# Very less p value

# 2) Taking care of seasonality along with trends

# Using time lag
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)
test_adFuller(ts_log_diff)

# ARIMA -> AR I MA 
# p value comes from PACF; q values comes from ACF
# P from AR; q from MA
# d from I

"""
# Split the data for ARIMA model
train = ts_log_diff[:120]
test = ts_log_diff[120:]
"""

train = ts_log_exp_diff[:120]
test = ts_log_exp_diff[120:]

# Build ARIMA model
import statsmodels.api as sm
model = sm.tsa.arima.ARIMA(train, order=(7,0,7)).fit()
pred = model.predict(start=len(train), end=(len(ts_log_exp_diff)-1))

# Model Evaluation
from sklearn.metrics import mean_squared_error

error = np.sqrt(mean_squared_error(test, pred))

test.mean(), np.sqrt(test.var())

train.plot(legend=True, label='Train', figsize=(10,6))
test.plot(legend=True, label='Test')
pred.plot(legend=True, label='predicted ARIMA')


# Predict the future data
final_model = sm.tsa.arima.ARIMA(ts, order=(7,0,7)).fit()
prediction = final_model.predict(len(ts), 
                                 len(ts)+36)

ts.plot(legend=True, label='Train')
prediction.plot(legend=True, label='Prediction')

# HOW TO GET HYPERPARAMTERS P,Q,D
# 1) ACF PACF plots
# 2) AUTO_ARIMA FUNCTION
# 3) Using custom for loops

import itertools

p = range(0,8)
q = range(0,8)
d = range(0,2)

pdq_combination = list(itertools.product(p,d,q))

rmse = []
order1 = []

for pdq in pdq_combination:
    try:
        model = sm.tsa.arima.ARIMA(train, order=pdq).fit()
        pred = model.predict(start=len(train), end=len(ts_log_diff)-1)
        error = np.sqrt(mean_squared_error(test, pred))
        order1.append(pdq)
        rmse.append(error)
        
    except:
        continue

results = pd.DataFrame(index=order1, data=rmse, columns=['RMSE'])
results.to_csv('ARIMA_results.csv')
# least rmse - 707
