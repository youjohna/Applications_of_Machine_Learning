# Jonathan You
# ITP 449 Summer 2022
# HW6
# Q2

# Converting a CSV into a dataframe into various graphs, as well as utilizing a linear regression machine learning model.

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

myDF = pd.read_csv('CommuteStLouis.csv')

# Question 1
print(myDF.describe())

# Question 2
print()
print(myDF[['Age', 'Distance', 'Time']].corr())
# The two numeric variables that are the most highly correlated are Time and Distance. The correlation coefficient is 0.830241, or 83.0241%

fig1 = plt.figure()
sb.pairplot(myDF[['Age', 'Distance', 'Time']])
plt.savefig('Question2-1.png')
# The figures in the diagonal show the relation between Age and Age, Distance and Distance, and Time and Time.
# The skewness of the various attributes imply that the majority of the distance traveled is on the shorter side, and also the time traveled is on the shorter side

fig2 = plt.figure()
sb.boxplot(data=myDF, x='Sex', y='Distance')
# Compared side by side with the boxplot for men, the boxplot for women show that the median of commute distance is slightly lower.
# However, one of the outliers for the women boxplot is significantly farther than all the men's commuting distance.

plt.savefig('Question2-2.png')

# Question 3
fig3 = plt.figure()
x = myDF[['Distance']]
y = myDF['Time']
model = LinearRegression()
model.fit(x.values, y.values)
xLine = np.linspace(0, 80, num=500).reshape(-11, 1)
yLine = model.predict(xLine)

plt.scatter(x, y)
plt.plot(xLine, yLine, 'b-')
plt.ylabel('Time')
plt.xlabel('Distance')
plt.savefig('Question2-3.png')
plt.title('Scatterplot and Linear Regression of Time vs Distance')

# Question 4
fig4 = plt.figure()
model = LinearRegression()
model.fit(x.values, y.values)
vis = ResidualsPlot(model)
vis.fit(x.values, y.values)
vis.show()
plt.savefig('Question2-4.png')

plt.show()