# Jonathan You
# ITP 449 Summer 2022
# HW6
# Q2

# Converting a CSV into a dataframe into various illustrating graphs

import pandas as pd
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# 1 Preparation
avocado = pd.read_csv('avocado.csv', usecols=['Date', 'AveragePrice', 'Total Volume'])
avocado['Date']=pd.to_datetime(avocado['Date'])
print(avocado)

# 2 Plotting
avocado.sort_values(by='Date', ascending=True, inplace=True)
myFig = plt.figure(1, figsize=(9, 7))
myFig.suptitle('Avocado Prices and Volume Time Series')

myAx1 = myFig.add_subplot(2, 2, 1)
myAx2 = myFig.add_subplot(2, 2, 2)
myAx3 = myFig.add_subplot(2, 2, 3, sharex=myAx1)
myAx4 = myFig.add_subplot(2, 2, 4, sharex=myAx2)

myAx1.scatter(avocado['Date'], avocado['AveragePrice'], s=5)
myAx1.set_ylabel('Average Price')
myAx2.scatter(avocado['Date'], avocado['Total Volume'], s=5)
myAx2.set_ylabel('Total Volume')

avocado['TotalRevenue'] = avocado['AveragePrice']*avocado['Total Volume']
avocado1 = avocado.groupby('Date').sum()
avocado1['AveragePrice'] = avocado1['TotalRevenue']/avocado1['Total Volume']
print(avocado1)

myAx3.plot(avocado1['AveragePrice'])
myAx3.tick_params(axis='x', labelrotation = 90)
myAx3.set_ylabel('Average Price')
myAx4.plot(avocado1['Total Volume'])
myAx4.tick_params(axis='x', labelrotation = 90)
myAx4.set_ylabel('Total Volume')

plt.setp(myAx1.get_xticklabels(), visible=False)
plt.setp(myAx2.get_xticklabels(), visible=False)


# 3 Plotting
myFig2 = plt.figure(2, figsize=(9,7))
myFig2.suptitle('Avocado Prices and Volume Time Series (Smoothed)')

myAx5 = myFig2.add_subplot(121)
myAx6 = myFig2.add_subplot(122)

myAx5.plot(avocado1['AveragePrice'].rolling(20).mean(), marker='.', markersize=5)
myAx5.tick_params(axis='x', labelrotation = 90)
myAx5.set_ylabel('Average Price')
myAx6.plot(avocado1['Total Volume'].rolling(20).mean(), marker='.', markersize=5)
myAx6.tick_params(axis='x', labelrotation = 90)
myAx6.set_ylabel('Total Volume')

plt.show()
