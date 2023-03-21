import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# A
myDF = pd.read_csv('Stores.csv')
pd.set_option('display.max_columns', None)
store = myDF['Store']
myDF.drop(columns='Store', inplace=True)

stdScaler = StandardScaler()
stdScaler.fit(myDF)
featureNorm = pd.DataFrame(stdScaler.transform(myDF), columns=myDF.columns)

# B
numCls = range(1, 51)
inertiaList = []

for n in numCls:
    # KMeans
    model3 = KMeans(n_clusters=n)
    model3.fit(featureNorm)
    inertiaList.append(model3.inertia_)

# C
plt.plot(numCls, inertiaList)
plt.xticks(numCls)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# D
# 5 is the best k

# E
model3 = KMeans(n_clusters=5)           # to control the random, add random_state=#
model3.fit(featureNorm)
print("The store belongs to cluster: ", model3.predict([[6.3,3.5,2.4,0.5]]))

# F
myDF['store'] = store
myDF['cluster'] = model3.labels_

# G
#x = np.arange(1, 6)
plt.figure()
plt.hist(myDF['cluster'])
plt.xlabel('Cluster Number')
plt.ylabel('Frequency')


#print(featureNorm)
plt.show()