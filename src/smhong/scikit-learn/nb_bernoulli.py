import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

def top_n_accuracy(train_data_pre, y_data, n):
    train_best_n = np.argsort(train_data_pre, axis=1)[:, -n:]
    count = 0
    for i in range(train_best_n.shape[0]):
        idx = y_data[i]
        if (train_best_n[i] == idx).any():
            count += 1
    return count / train_data_pre.shape[0] * 100


def get_classes_prob(train_data_pre, y_data, n):
    train_best_n = np.argsort(train_data_pre, axis=1)[:, -n:]
    total = np.zeros(11)
    correct = np.zeros(11)
    count = 0
    for i in range(train_best_n.shape[0]):
        idx = y_data[i]
        if (train_best_n[i] == idx).any():
            count += 1
            correct[idx] += 1
            total[idx] += 1
        else:
            total[idx] += 1
    i = 0
    for i in range(11):
        print('i =', i, ', total =', total[i], ', correct =', correct[i], ', prob =', correct[i] / total[i] * 100)

data = pd.read_csv("../../../data/smhong/data_ver3.csv")

data = shuffle(data)

data.info()

X = data.drop("Club_Position", axis=1)
X = X.drop("Club_Position1", axis=1)
Y = data["Club_Position1"]

X.head()
Y.head()

X = StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=43)

print("X_Train Shape :", np.shape(Y_train))
print("Y_Train Shape :", np.shape(Y_train))
print(Y_train.ravel())

nb = BernoulliNB()
nb.fit(X_train, Y_train)

train_data_pre = nb.predict(X_train)
train_acc_score = metrics.accuracy_score(Y_train.values, train_data_pre)
print("Train Data Accuracy :", train_acc_score)

train_data_pre = nb.predict_proba(X_train)
train_top_k_acc_score = top_n_accuracy(train_data_pre, Y_train.values, 3)
print("Train Data Top 3 Accuracy :", train_top_k_acc_score)

test_data_pre = nb.predict(X_test)
test_acc_score = metrics.accuracy_score(Y_test, test_data_pre)
print("Test Data Accuracy :", test_acc_score)

test_data_pre = nb.predict_proba(X_test)
test_top_k_acc_score = top_n_accuracy(test_data_pre, Y_test.values, 3)
print("Test Data Top 3 Accuracy :", test_top_k_acc_score)

get_classes_prob(test_data_pre, Y_test.values, 3)

'''
Train Data Accuracy : 0.4684408389394539
Train Data Top 3 Accuracy : 77.3644637910566
Test Data Accuracy : 0.47229916897506924
Test Data Top 3 Accuracy : 78.30101569713757
i = 0 , total = 462.0 , correct = 462.0 , prob = 100.0
i = 1 , total = 530.0 , correct = 442.0 , prob = 83.39622641509435
i = 2 , total = 364.0 , correct = 341.0 , prob = 93.68131868131869
i = 3 , total = 581.0 , correct = 487.0 , prob = 83.8209982788296
i = 4 , total = 448.0 , correct = 304.0 , prob = 67.85714285714286
i = 5 , total = 704.0 , correct = 471.0 , prob = 66.9034090909091
i = 6 , total = 449.0 , correct = 213.0 , prob = 47.43875278396437
i = 7 , total = 140.0 , correct = 119.0 , prob = 85.0
i = 8 , total = 8.0 , correct = 2.0 , prob = 25.0
i = 9 , total = 147.0 , correct = 121.0 , prob = 82.31292517006803
i = 10 , total = 499.0 , correct = 430.0 , prob = 86.17234468937876
'''