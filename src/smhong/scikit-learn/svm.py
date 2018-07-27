import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import svm, metrics

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
Train Data Accuracy : 0.6521567075583696
Train Data Top 3 Accuracy : 93.96517609814009
Test Data Accuracy : 0.5997229916897507
Test Data Top 3 Accuracy : 90.81255771006464
i = 0 , total = 456.0 , correct = 456.0 , prob = 100.0
i = 1 , total = 493.0 , correct = 471.0 , prob = 95.53752535496957
i = 2 , total = 375.0 , correct = 348.0 , prob = 92.80000000000001
i = 3 , total = 563.0 , correct = 534.0 , prob = 94.84902309058614
i = 4 , total = 450.0 , correct = 399.0 , prob = 88.66666666666667
i = 5 , total = 721.0 , correct = 690.0 , prob = 95.7004160887656
i = 6 , total = 460.0 , correct = 418.0 , prob = 90.8695652173913
i = 7 , total = 151.0 , correct = 70.0 , prob = 46.35761589403973
i = 8 , total = 19.0 , correct = 0.0 , prob = 0.0
i = 9 , total = 146.0 , correct = 81.0 , prob = 55.47945205479452
i = 10 , total = 498.0 , correct = 467.0 , prob = 93.77510040160642
'''