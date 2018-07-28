import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import metrics, model_selection
from sklearn.naive_bayes import GaussianNB

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

nb = GaussianNB()
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
Train Data Accuracy : 0.4881282152750297
Train Data Top 3 Accuracy : 80.32251681836169
Test Data Accuracy : 0.4778393351800554
Test Data Top 3 Accuracy : 79.3859649122807
i = 0 , total = 477.0 , correct = 477.0 , prob = 100.0
i = 1 , total = 514.0 , correct = 443.0 , prob = 86.18677042801556
i = 2 , total = 338.0 , correct = 312.0 , prob = 92.3076923076923
i = 3 , total = 548.0 , correct = 472.0 , prob = 86.13138686131386
i = 4 , total = 463.0 , correct = 296.0 , prob = 63.93088552915766
i = 5 , total = 715.0 , correct = 525.0 , prob = 73.42657342657343
i = 6 , total = 459.0 , correct = 233.0 , prob = 50.76252723311547
i = 7 , total = 133.0 , correct = 102.0 , prob = 76.69172932330827
i = 8 , total = 9.0 , correct = 1.0 , prob = 11.11111111111111
i = 9 , total = 147.0 , correct = 120.0 , prob = 81.63265306122449
i = 10 , total = 529.0 , correct = 458.0 , prob = 86.57844990548205
'''