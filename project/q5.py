import pandas as pd
import os as os
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

cwd = '/Users/Jonathan/Desktop/ITP 449/HOMEWORK/project'
os.chdir(cwd)

myDF = pd.read_csv('auto-mpg.csv')
pd.set_option('display.max_columns', None)

# A
print(myDF.describe())
# the mean of mpg is 23.514573

# B
print('\nThe median of mpg is: ', myDF['mpg'].median())

# C
# The mean is slightly higher. This indicates that the graph is right skewed
plt.hist(myDF['mpg'], bins=15)

# D
sb.pairplot(myDF.drop(columns='No'))
plt.savefig('q5_pairplot.png')

# E
# Weight and Horsepower seems to be the most strongly linearly correlated

# F
# Acceleration and Weight seem to be the most weakly correlated

# G
plt.figure()
plt.scatter(myDF['displacement'], myDF['mpg'])

# H
plt.figure()
model = LinearRegression()
model.fit(myDF[['displacement']], myDF[['mpg']])
print(model.coef_)          # slope
print(model.intercept_)     # y-intercept
# y = -0.06028241x + 35.17475015
# The model predicts that mpg will decrease as displacement increases
# given a car with displacement value of 200, the predicted mpg is 23.1182681

plt.scatter(myDF[['displacement']], myDF[['mpg']])
xLine = np.arange(1,501).reshape(-1, 1)
yLine = model.predict(xLine)
plt.plot(xLine, yLine, 'r-')

plt.figure()
vis = ResidualsPlot(model)
vis.fit(myDF[['displacement']], myDF[['mpg']])
vis.show()

plt.show()