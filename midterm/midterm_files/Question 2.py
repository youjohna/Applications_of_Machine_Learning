import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Question 1
aaplDF = pd.read_csv('AAPL(3).csv')

# Question 2
aaplDF['Date'] = pd.to_datetime(aaplDF['Date'])
aaplDF.set_index('Date', inplace=True)

# Question 3
high = (aaplDF.sort_values(by='High', ascending=False)[0:1]['High'])
low = (aaplDF.sort_values(by='High', ascending=True)[0:1]['High'])

print(high)
print(low)
# Question 4
print()
print(high.index-low.index)

# Question 5
print()
print(aaplDF.loc[high.index, 'Volume'])

# Question 6
fig = plt.figure(figsize=(9,7))
plt.plot(aaplDF['Close'])
plt.xticks(rotation='vertical')
plt.title('Apple Stock vs Time')
plt.ylabel('Stock Prices')

# Question 7
aaplDF['Close'].rolling(20).mean().plot()
plt.legend(['Daily','Avg'], loc='best')

plt.show()