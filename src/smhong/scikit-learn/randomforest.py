import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import metrics, model_selection
from sklearn.ensemble import RandomForestClassifier

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

clf = RandomForestClassifier()
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
Train Data Accuracy : 0.9893153937475268
Train Data Top 3 Accuracy : 99.99010684606252
Test Data Accuracy : 0.5431671283471837
Test Data Top 3 Accuracy : 82.29455216989842
i = 0 , total = 463.0 , correct = 463.0 , prob = 100.0
i = 1 , total = 512.0 , correct = 445.0 , prob = 86.9140625
i = 2 , total = 352.0 , correct = 297.0 , prob = 84.375
i = 3 , total = 576.0 , correct = 523.0 , prob = 90.79861111111111
i = 4 , total = 480.0 , correct = 290.0 , prob = 60.416666666666664
i = 5 , total = 701.0 , correct = 606.0 , prob = 86.44793152639087
i = 6 , total = 467.0 , correct = 335.0 , prob = 71.73447537473233
i = 7 , total = 139.0 , correct = 62.0 , prob = 44.60431654676259
i = 8 , total = 12.0 , correct = 0.0 , prob = 0.0
i = 9 , total = 147.0 , correct = 82.0 , prob = 55.78231292517006
i = 10 , total = 483.0 , correct = 462.0 , prob = 95.65217391304348
'''