import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# A
myDF = pd.read_csv('winequality(1).csv')
pd.set_option('display.max_columns', None)

feature = myDF.drop(columns='Quality')
target = myDF['Quality']

# B
stdScaler = StandardScaler()
stdScaler.fit(feature)
featureNorm = pd.DataFrame(stdScaler.transform(feature), columns=feature.columns)

# C
X_train, X_test, y_train, y_test = train_test_split(featureNorm, target, train_size=0.6, random_state=2022, stratify=target)
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train, train_size=0.5, random_state=2022, stratify=y_train)

# D and E
k_range = np.arange(1, 30)
trainA_accuracy = []
trainB_accuracy = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_trainA, y_trainA)
    y_predA = model.predict(X_trainA)
    y_predB = model.predict(X_trainB)
    trainA_accuracy.append(metrics.accuracy_score(y_trainA, y_predA))
    trainB_accuracy.append(metrics.accuracy_score(y_trainB, y_predB))

# F
plt.plot(k_range, trainA_accuracy, label='TrainA', color='b')
plt.plot(k_range, trainB_accuracy, label='TrainB', color='r')
plt.xticks(k_range)
plt.title('Accuracy of Train A and Train B')
plt.legend()
plt.xlabel('k values')
plt.ylabel('Accuracy')
# Value K produced the best accuracy for both train A and train B

# G
model = KNeighborsClassifier(n_neighbors=12)
model.fit(X_trainA, y_trainA)
y_pred_test = model.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred_test))
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)

# H
feature['Quality'] = target
feature['Predicted Quality'] = model.predict(featureNorm)
print(feature)

# I
print('\nAccuracy for test set with k=9: ', metrics.accuracy_score(y_test, y_pred_test))
plt.show()