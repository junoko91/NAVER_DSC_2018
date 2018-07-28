import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import metrics, neighbors

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

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, Y_train)

train_data_pre = knn.predict(X_train)
train_acc_score = metrics.accuracy_score(Y_train.values, train_data_pre)
print("Train Data Accuracy :", train_acc_score)

train_data_pre = knn.predict_proba(X_train)
train_top_k_acc_score = top_n_accuracy(train_data_pre, Y_train.values, 3)
print("Train Data Top 3 Accuracy :", train_top_k_acc_score)

test_data_pre = knn.predict(X_test)
test_acc_score = metrics.accuracy_score(Y_test, test_data_pre)
print("Test Data Accuracy :", test_acc_score)

test_data_pre = knn.predict_proba(X_test)
test_top_k_acc_score = top_n_accuracy(test_data_pre, Y_test.values, 3)
print("Test Data Top 3 Accuracy :", test_top_k_acc_score)

get_classes_prob(test_data_pre, Y_test.values, 3)

'''
Train Data Accuracy : 0.6672932330827067
Train Data Top 3 Accuracy : 96.46814404432132
Test Data Accuracy : 0.5279316712834718
Test Data Top 3 Accuracy : 78.57802400738689
i = 0 , total = 457.0 , correct = 457.0 , prob = 100.0
i = 1 , total = 578.0 , correct = 453.0 , prob = 78.37370242214533
i = 2 , total = 375.0 , correct = 322.0 , prob = 85.86666666666667
i = 3 , total = 536.0 , correct = 454.0 , prob = 84.70149253731343
i = 4 , total = 455.0 , correct = 243.0 , prob = 53.40659340659341
i = 5 , total = 724.0 , correct = 602.0 , prob = 83.14917127071824
i = 6 , total = 436.0 , correct = 284.0 , prob = 65.13761467889908
i = 7 , total = 143.0 , correct = 51.0 , prob = 35.66433566433567
i = 8 , total = 8.0 , correct = 0.0 , prob = 0.0
i = 9 , total = 146.0 , correct = 86.0 , prob = 58.9041095890411
i = 10 , total = 474.0 , correct = 452.0 , prob = 95.35864978902954
'''