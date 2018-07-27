import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import svm, metrics, model_selection

data = pd.read_csv("../../../data/smhong/data_ver3.csv")
data = shuffle(data)

X = data.drop("Club_Position", axis=1)
X = X.drop("Club_Position1", axis=1)
Y = data["Club_Position1"]

nbclass = len(Y.unique())

print(nbclass)

X.head()
Y.head()

X = StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=43)

print("X_Train Shape :", np.shape(X_train))
print("Y_Train Shape :", np.shape(Y_train))

clf = svm.SVC(probability=True)
clf.fit(X_train, Y_train)

train_data_pre = clf.predict(X_train)

train_acc_score = metrics.accuracy_score(Y_train, train_data_pre)
print("Train Data Accuracy :", train_acc_score)

test_data_pre = clf.predict(X_test)

test_acc_score = metrics.accuracy_score(Y_test, test_data_pre)
print("Test Data Accuracy :", test_acc_score)

test_cross_val_scores = model_selection.cross_val_score(clf, X, Y, cv=5)
print("Cross Val Score Accuracy :", test_cross_val_scores.mean())
