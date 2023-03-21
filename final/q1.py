import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Question 1
myDF = pd.read_csv('wineQualityReds(2).csv')
pd.set_option('display.max_columns', None)

# Question 2
print(myDF.shape)
print()

# Question 3
for col in myDF.columns:
    print(col,' has ', sum(np.isnan(myDF[col])), " null values")

# Question 4
target = myDF['quality']
feature = myDF.drop(columns=['quality', 'Wine'])

# Question 5
stdScaler = StandardScaler()
stdScaler.fit(feature)
featureNorm = pd.DataFrame(stdScaler.transform(feature), columns=feature.columns)

# Question 6
X_train, X_test, y_train, y_test = train_test_split(featureNorm, target, test_size=0.25, random_state=2022, stratify=target)
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_train, y_train, test_size=0.33, random_state=2022, stratify=y_train)

# Question 7
print('\nThere are ',X_train.size,' cases in the Train partition')

# Question 8
print('There are ',X_test.size,' cases in the Train partition')

# Question 9 and 10
k_range = np.arange(1,30)
trainA_accuracy = []
trainB_accuracy = []
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_trainA, y_trainA)
    y_predA = model.predict(X_trainA)
    y_predB = model.predict(X_trainB)
    trainA_accuracy.append(metrics.accuracy_score(y_trainA, y_predA))
    trainB_accuracy.append(metrics.accuracy_score(y_trainB, y_predB))

# Question 11
plt.plot(k_range, trainA_accuracy, label='TrainA', color='b')
plt.plot(k_range, trainB_accuracy, label='TrainB', color='r')
plt.xticks(k_range)
plt.title('Accuracy of Train A and Train B')
plt.legend()
plt.xlabel('k values')
plt.ylabel('Accuracy')

# Question 12
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_trainA, y_trainA)
y_pred_test = model.predict(X_test)
print('\nAccuracy for test set with k=9: ',metrics.accuracy_score(y_test, y_pred_test))

# Question 13
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)

# Question 14
y_pred_A = model.predict(X_trainA)
print('\nAccuracy for Train A partition:', metrics.accuracy_score(y_trainA, y_pred_A))

# Question 15
y_pred_B = model.predict(X_trainB)
print('\nAccuracy for Train B partition:', metrics.accuracy_score(y_trainB, y_pred_B))

# Question 16
sample_x = np.array([[8, 0.6, 0, 2.0, 0.067, 10, 30, 0.9978, 3.20, 0.5, 10.0]])
sample_y = model.predict(sample_x)
print('\nThe quality of your sample wine is: ',sample_y)

plt.show()
