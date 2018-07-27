import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import svm, metrics

def top_n_accuracy(train_data_pre, y_data, n):
    print(np.argsort(train_data_pre, axis=1)[0:5])
    train_best_n = np.argsort(train_data_pre, axis=1)[:, -n:]
    count = 0
    print(train_best_n[0:5])
    print(y_data[0:5])
    for i in range(train_best_n.shape[0]):
        idx = y_data[i]
        if (train_best_n[i] == idx).any() :
            #print('i :', i, ', idx :', idx, ', x_data[i, idx] :', train_best_n[i])
            count += 1
    return count / train_data_pre.shape[0] * 100


data = pd.read_csv("../../../data/smhong/data_ver3.csv")
data = shuffle(data)

X = data.drop("Club_Position", axis=1)
X = X.drop("Club_Position1", axis=1)
y = data["Club_Position1"]

nbclass = len(y.unique())

print(nbclass)

X.head()
y.head()

X = StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=43)

print("X_Train Shape :", np.shape(X_train))
print("Y_Train Shape :", np.shape(Y_train))

clf = svm.SVC(probability=True)
clf.fit(X_train, Y_train)
train_data_pre = clf.predict_proba(X_train)

train_acc_score = top_n_accuracy(train_data_pre, Y_train.values, 1)

print("Train Data Top 3 Accuracy :", train_acc_score)

train_data_pre = clf.predict(X_train)
print(train_data_pre[0:5])
train_acc_score = metrics.accuracy_score(Y_train.values, train_data_pre)
print("Train Data Accuracy :", train_acc_score)

'''
train_acc_score = metrics.accuracy_score(Y_train, train_data_pre)
print("Train Data Accuracy :", train_acc_score)

test_data_pre = clf.predict(X_test)
test_acc_score = metrics.accuracy_score(Y_test, test_data_pre)
print("Test Data Accuracy :", test_acc_score)
'''
