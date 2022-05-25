# Time-Series-Forcasting---AirPassengers
The dataset consists of 2 columns - months and no. of passengers travelling in that month. A classic time-series ML problem!

The process --> 
1) Convert the month column's data type from object to datetime. Place it as the index of the dataframe.
2) A simple plt.plot() shows the trends and seasonality of the data.
3) The data needs to be stationary over time ->
    - The augumented Dicky-Fuller (ADF) test will present the statistics with respect to p-values.
    - If p > 0.05; Non-stationary || p <= 0.05; stationary
    - This dataset turns out to have p value greater than 0.05.
4) Taking care of the trends and seasonality
5) Smoothing the trend in the dataset ->
    - Getting the rolling mean and subtracting it from the original data -> testing ADF -> lower p value
    - Exponentialy weighted mean with halflife -> subtracting result from data -> testing ADF -> even lower p value
6) Applying ARIMA model on the data
7) A manual for loop to obtain the lowest RMSE for the P,Q,D values for the above ARIMA model.
