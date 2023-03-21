# Jonathan You
# ITP 449 Summer 2022
# HW6
# Q2

# Converting a CSV into a dataframe and then performing a logistical regression model on the data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics

# Question 1
myDF = pd.read_csv('Titanic.csv')

# Question 2
# Target variable is 'Survived'. Two possible values are No and Yes
# Requires logistic regression model, categorical

# Question 3
myDF = myDF.drop('Passenger', axis=1)

# Question 4
print(myDF.isnull().any())  # True if there is at least ONE null value
# If any are true, then null values exist

# Question 5
fig, axes = plt.subplots(2,2)
plt.suptitle('Count Plots of Remaining Factors')
sb.countplot(data=myDF, y='Class', ax=axes[0,0])
sb.countplot(data=myDF, y='Sex', ax=axes[0,1])
sb.countplot(data=myDF, y='Age', ax=axes[1,0])

# Question 6
X = myDF[['Class', 'Sex', 'Age']]
y = myDF['Survived']
X_dummy = pd.get_dummies(X, drop_first = True)      # drop_first not required, but it's technically not needed
print(X_dummy)

# Question 7
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.25, random_state=2022)

# Question 8
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Question 9
print(metrics.classification_report(y_test, y_pred))

# Question 10
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.suptitle('Confusion Matrix')

# Question 11
sample_X = np.array([[1,0,0,0,0]])
sample_y_pred = model.predict(sample_X)
print(sample_y_pred)

plt.show()
