import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics

myDF = pd.read_csv('mushrooms(1).csv')
pd.set_option('display.max_columns', None)

feature = myDF.drop(columns='class')
target = myDF['class']


X_train, X_test, y_train, y_test = train_test_split(feature, target, train_size=0.75, random_state=2022, stratify=target)
plt.figure(figsize=(12, 7))
model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=2022)
model.fit(X_train, y_train)