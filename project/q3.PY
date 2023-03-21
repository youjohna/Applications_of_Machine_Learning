import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics

myDF = pd.read_csv('UniversalBank(1).csv')
pd.set_option('display.max_columns', None)

# A
# The target variable is 'Personal Loan'

# B
target = myDF['Personal Loan']
feature = myDF.drop(columns=['Personal Loan', 'Row', 'ZIP Code'])

# C
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.75, random_state=2022, stratify=target)

# D
print(sum(y_train==1), ' cases of the training partition represented people who accepted the offer or a personal loan')

# E
plt.figure(figsize=(12, 7))
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=2022)
model.fit(X_train, y_train)
tree.plot_tree(model, feature_names=X_train.columns, class_names='Personal Loan', filled=True, fontsize=6)

# F
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
print("There are 42 acceptors that the model classified as non-acceptors")

# G
print("There are 50 acceptors that the model classified as non-acceptors")

# H
print('Accuracy for training partition: ', metrics.accuracy_score(y_test, y_pred_test))
print('Accuracy for testing partition: ', metrics.accuracy_score(y_train, y_pred_train))

plt.show()