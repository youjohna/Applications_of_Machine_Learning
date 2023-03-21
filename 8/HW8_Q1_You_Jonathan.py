# Jonathan You
# ITP 449 Summer 2022
# HW8
# Q1


import pandas as pd
import numpy as np
import os as os
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Question 1
myDF = pd.read_csv('wineQualityReds(1).csv')

# Question 2
myDF.drop(columns='Wine', inplace=True)

# Question 3
qualityArray = myDF['quality']

# Question 4
myDF.drop(columns='quality', inplace=True)

# Question 5
print(myDF)
print(qualityArray)

# Question 6
mmScaler = MinMaxScaler()
mmScaler.fit(myDF)
myDFNorm = pd.DataFrame(mmScaler.transform(myDF), columns=myDF.columns)

# Question 7
print(myDFNorm)

# Question 8
numCls = range(1, 21)
inertiaList = []

for n in numCls:
    # KMeans
    model3 = KMeans(n_clusters=n)           # As number of clusters go up, inertia (within-cluster sum of squares) drops
    model3.fit(myDFNorm)
    inertiaList.append(model3.inertia_)

# Question 9
plt.plot(numCls, inertiaList, marker='.', markersize=10)
plt.xlabel('Number of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(numCls)


# Question 10
# I would choose 6 or 7 as the number of clusters

# Question 11
model = KMeans(n_clusters=6, random_state=2022)
model.fit(myDFNorm)
myDF['clusters'] = (model.labels_)
print(myDF)

# Question 12
myDF['quality'] = qualityArray

# Question 13
print(pd.crosstab(index=myDF['quality'], columns=myDF['clusters']))
# The clusters do not represent the quality of wine


plt.show()