import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics

myDF = pd.read_csv('mushrooms(1).csv')
pd.set_option('display.max_columns', None)

feature = myDF.drop(columns='class')
X_dummy = pd.get_dummies(feature[:])
target = myDF['class']
y_dummy = pd.get_dummies(target)


X_train, X_test, y_train, y_test = train_test_split(X_dummy, target, train_size=0.75, random_state=2022, stratify=target)
plt.figure(figsize=(12, 7))
model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=2022)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


# A
print(metrics.confusion_matrix(y_test, y_pred_test))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)

# B
print('Accuracy for testing partition: ', metrics.accuracy_score(y_train, y_pred_train))

# C
print('Accuracy for training partition: ', metrics.accuracy_score(y_test, y_pred_test))

# D
plt.figure()
tree.plot_tree(model, feature_names=X_train.columns, class_names='Personal Loan', filled=True, fontsize=6)

# E
# odor_n, bruises_f, spore_print_color

# F


plt.show()
