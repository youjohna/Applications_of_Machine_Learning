# Jonathan You
# ITP 449 Summer 2022
# HW5
# Question 2
# Created dataframes from three csvs to create graphs

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

myDF1 = pd.read_csv('time_series_covid19_confirmed_US.csv')
myDF2 = pd.read_csv('time_series_covid19_deaths_US.csv')
myDF3 = pd.read_csv('10-02-2021.csv')

# Question 1
print(myDF2.groupby(by='Province_State').sum().sort_values(by='10/2/21', ascending=False)[0:1]['10/2/21'])

# Question 2
print()
print(myDF1.groupby(by='Province_State').sum().sort_values(by='10/2/21', ascending=False)[1:2]['10/2/21'])

# Question 3
print()
print(myDF3['Testing_Rate'].max() - myDF3['Testing_Rate'].min())

# Question 4
# top5 = (myDF1.groupby(by='Province_State').sum().sort_values(by='10/2/21', ascending=False)[0:5]['10/2/21'])
myDF1trans = (myDF1.groupby(by='Province_State').sum().sort_values(by='10/2/21', ascending=False)[0:5])
myDF1trans.drop(myDF1trans.iloc[:, 0:10], axis=1, inplace=True)
myDF1trans = myDF1trans.transpose()
#print(myDF1trans)
myFig = plt.figure(1)
myDF1trans['California'].plot()
myDF1trans['Texas'].plot()
myDF1trans['Florida'].plot()
myDF1trans['New York'].plot()
myDF1trans['Illinois'].plot()
plt.title('New Cases over Time')
plt.legend(loc='best')

# Question 5
myDF2trans = (myDF2.groupby(by='Province_State').sum().sort_values(by='10/2/21', ascending=False).loc[['California','Texas','Florida','New York','Illinois']])
myDF2trans.drop(myDF2trans.iloc[:, 0:11], axis=1, inplace=True)
myDF2trans = myDF2trans.transpose()
myFig = plt.figure(2)
myDF2trans['California'].plot()
myDF2trans['Texas'].plot()
myDF2trans['Florida'].plot()
myDF2trans['New York'].plot()
myDF2trans['Illinois'].plot()
plt.title('Deaths over Time')
plt.legend(loc='best')

plt.show()
