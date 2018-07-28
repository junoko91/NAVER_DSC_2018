import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import tree, metrics


def top_n_accuracy(train_data_pre, y_data, n):
    train_best_n = np.argsort(train_data_pre, axis=1)[:, -n:]
    count = 0
    # print(train_best_n[0:5])
    # print(y_data[0:5])
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

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)

train_data_pre = clf.predict(X_train)
train_acc_score = metrics.accuracy_score(Y_train.values, train_data_pre)
print("Train Data Accuracy :", train_acc_score)

train_data_pre = clf.predict_proba(X_train)
train_top_k_acc_score = top_n_accuracy(train_data_pre, Y_train.values, 3)
print("Train Data Top 3 Accuracy :", train_top_k_acc_score)

test_data_pre = clf.predict(X_test)
test_acc_score = metrics.accuracy_score(Y_test, test_data_pre)
print("Test Data Accuracy :", test_acc_score)

test_data_pre = clf.predict_proba(X_test)
test_top_k_acc_score = top_n_accuracy(test_data_pre, Y_test.values, 3)
print("Test Data Top 3 Accuracy :", test_top_k_acc_score)

get_classes_prob(test_data_pre, Y_test.values, 3)

'''
Train Data Accuracy : 1.0
Train Data Top 3 Accuracy : 100.0
Test Data Accuracy : 0.4669898430286242
Test Data Top 3 Accuracy : 54.77839335180056
i = 0 , total = 445.0 , correct = 445.0 , prob = 100.0
i = 1 , total = 510.0 , correct = 278.0 , prob = 54.509803921568626
i = 2 , total = 368.0 , correct = 163.0 , prob = 44.29347826086957
i = 3 , total = 560.0 , correct = 283.0 , prob = 50.535714285714285
i = 4 , total = 476.0 , correct = 107.0 , prob = 22.478991596638657
i = 5 , total = 685.0 , correct = 306.0 , prob = 44.67153284671533
i = 6 , total = 488.0 , correct = 107.0 , prob = 21.92622950819672
i = 7 , total = 125.0 , correct = 14.0 , prob = 11.200000000000001
i = 8 , total = 9.0 , correct = 4.0 , prob = 44.44444444444444
i = 9 , total = 156.0 , correct = 156.0 , prob = 100.0
i = 10 , total = 510.0 , correct = 510.0 , prob = 100.0
'''
