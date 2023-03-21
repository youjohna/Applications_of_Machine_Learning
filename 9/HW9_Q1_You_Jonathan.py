# Jonathan You
# ITP 449 Summer 2022
# HW9
# Q1

import pandas as pd
import numpy as np
import os as os
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Question 1
myDF = pd.read_csv('diabetes(1).csv')
pd.set_option('display.max_columns', None)

# Question 2
print(myDF.shape)

# Question 3
for columns in myDF.columns:
    print(sum(np.isnan(myDF[columns])))
# no missing  values

# Question 4
myDFTarget = myDF['Outcome']
myDFFeature = myDF.drop(columns='Outcome')

# Question 5
stdScaler = StandardScaler()
stdScaler.fit(myDFFeature)
featureNorm = pd.DataFrame(stdScaler.transform(myDFFeature), columns=myDFFeature.columns)
print(featureNorm)

# Question 6
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(featureNorm, myDFTarget, test_size=0.3, random_state=2022, stratify=myDFTarget)

# Question 7
n_range = np.arange(1, 9)
accuracyListA = []
accuracyListB = []
for n in n_range:
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_trainA, y_trainA)
    y_predA = model.predict(X_trainA)
    y_predB = model.predict(X_trainB)
    accuracyListA.append(metrics.accuracy_score(y_trainA, y_predA))
    accuracyListB.append(metrics.accuracy_score(y_trainB, y_predB))

# Question 8
plt.plot(n_range, accuracyListA, label='TrainA')
plt.plot(n_range, accuracyListB, label='Train B')
plt.xlabel('# of neighbors (k value)')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(n_range)


# Question 9
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_trainA, y_trainA)
y_predB_test = model.predict((X_trainB))
print('Accuracy on the Train B set: ', metrics.accuracy_score(y_trainB, y_predB_test))
print('Confusion Matrix on Train B set:\n', metrics.confusion_matrix(y_trainB, y_predB_test))
metrics.ConfusionMatrixDisplay.from_predictions(y_trainB, y_predB_test)

# Question 10

outcome_values = np.array([[6, 140, 60, 12, 300, 24, 0.4, 45]])
outcome_values_transform = pd.DataFrame(stdScaler.transform(outcome_values))
outcome_prediction = model.predict(outcome_values_transform)
if outcome_prediction == 1:
    print("The model predicts that the person will have diabetes")
else:
    print("The model predicts that the person will NOT have diabetes")

plt.show()
