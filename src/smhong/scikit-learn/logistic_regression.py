import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import linear_model, metrics

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
Y = data["Club_Position1"]

nbclass = len(Y.unique())

print(nbclass)

X.head()
Y.head()

X = StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=43)

print("X_Train Shape :", np.shape(X_train))
print("Y_Train Shape :", np.shape(Y_train))

clf = linear_model.LogisticRegression()
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
i = 0 , total = 460.0 , correct = 460.0 , prob = 100.0
i = 1 , total = 518.0 , correct = 479.0 , prob = 92.47104247104248
i = 2 , total = 358.0 , correct = 346.0 , prob = 96.64804469273743
i = 3 , total = 570.0 , correct = 542.0 , prob = 95.08771929824562
i = 4 , total = 407.0 , correct = 340.0 , prob = 83.53808353808354
i = 5 , total = 751.0 , correct = 706.0 , prob = 94.00798934753661
i = 6 , total = 446.0 , correct = 368.0 , prob = 82.51121076233184
i = 7 , total = 141.0 , correct = 58.0 , prob = 41.13475177304964
i = 8 , total = 11.0 , correct = 0.0 , prob = 0.0
i = 9 , total = 137.0 , correct = 61.0 , prob = 44.52554744525548
i = 10 , total = 533.0 , correct = 515.0 , prob = 96.62288930581614
'''