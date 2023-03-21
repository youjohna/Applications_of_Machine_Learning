import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Question 1
myDF = pd.read_csv('Breast_Cancer(1).csv')
pd.set_option('display.max_columns', None)

# Question 2
for col in myDF.columns:
    print(col, ' has ', sum(myDF[col].isnull()), ' null values')

# Question 3
target = myDF['diagnosis']
feature = myDF.drop(columns='diagnosis')

# Question 4
sb.countplot(target)
plt.title('Countplot of Diagnosis')

# Question 5
X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state = 2022, stratify=target)

# Question 6
model = LogisticRegression()
model.fit(X_train, y_train)

# Question 7
y_pred = model.predict(X_train)
print(metrics.classification_report(y_train, y_pred))

# Question 8
print(metrics.confusion_matrix(y_train, y_pred))
metrics.ConfusionMatrixDisplay.from_predictions(y_train, y_pred)
plt.title('Confusion Matrix')

plt.show()


#print(myDF)