import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime

# Plot 1
plt.subplot(3, 1, 1)
x = np.random.randint(low=1, high=500, size=100)
y = np.random.normal(size=100)
plt.scatter(x, y, s=20, color='g')
plt.ylabel('Random Normal')
plt.title('Scatter Chart')

# Plot 2
plt.subplot(3, 1, 2)
A = np.arange(0, 12, 0.01)
B = np.sin(np.arange(0, 12, 0.01))
plt.yticks(np.arange(-1,2,0.5))
plt.xlim(-1,11)
plt.plot(A, B)
plt.ylabel('SINE(X)')
plt.title('Sine')

# Plot 3
date = [datetime(2020,2,1),
        datetime(2020,5,1),
        datetime(2020,8,1),
        datetime(2020,11,1),
        datetime(2021,2,1),
        datetime(2021,5,1),
        datetime(2021,8,1)]
plt.subplot(3,1,3)
myDF = pd.read_csv('time_series_covid19(1).csv')
myDF['Date'] = pd.to_datetime(myDF['Date'])
print(myDF['Date'])
myDF.set_index('Date', inplace=True)
plt.plot(myDF['US'], color='r', marker='.', markersize=10)
plt.xticks(date)
plt.yticks(np.arange(0,50000000,10000000))
plt.title('US Covid-19 Cases')
plt.ylabel('Cases')
plt.xlabel('Date')

plt.show()