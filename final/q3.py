import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree

# Question 1
ccDefaults = pd.read_csv('ccDefaults(1).csv')
pd.set_option('display.max_columns', None)

# Question 2
print(ccDefaults.info())
print()
for col in ccDefaults.columns:
    print(col, ' has ', sum(ccDefaults[col].isnull()), ' null values')

# Question 3
print()
print(ccDefaults.iloc[:5, :])

# Question 4
print()
print(ccDefaults.shape)

# Question 5
ccDefaults.drop(columns='ID', inplace=True)

# Question 6
print()
print('Previous Dimension is', ccDefaults.shape)
ccDefaults.drop_duplicates(keep='first', inplace=True)
print('New Dimension is', ccDefaults.shape)

# Question 7
print()
print(ccDefaults.corr())

# Question 8
target = ccDefaults['dpnm']
feature = ccDefaults[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']]

# Question 9
X_train, X_test, y_train, y_test = train_test_split(feature, target, random_state=2022, stratify=target)

# Question 10
model = DecisionTreeClassifier(criterion='entropy',max_depth=4, random_state=2022)
model.fit(X_train, y_train)

# Question 11
y_pred = model.predict(X_test)
print('The accuracy of the test partition is: ', metrics.accuracy_score(y_test, y_pred))

# Question 12
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# Question 13
plt.figure(figsize=(16, 7))
tree.plot_tree(model, feature_names=X_train.columns, class_names='dpnm', filled=True, fontsize=5)
plt.savefig('decisionTree.png')

plt.show()