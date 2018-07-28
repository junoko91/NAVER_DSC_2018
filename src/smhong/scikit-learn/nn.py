import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import metrics, model_selection
from sklearn.neural_network import MLPClassifier

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
'''
nn = clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(86, 86, 86), random_state=1)
'''
nn = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(86, 86, 86), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

nn.fit(X_train, Y_train)

train_data_pre = nn.predict(X_train)
train_acc_score = metrics.accuracy_score(Y_train.values, train_data_pre)
print("Train Data Accuracy :", train_acc_score)

train_data_pre = nn.predict_proba(X_train)
train_top_k_acc_score = top_n_accuracy(train_data_pre, Y_train.values, 3)
print("Train Data Top 3 Accuracy :", train_top_k_acc_score)

test_data_pre = nn.predict(X_test)
test_acc_score = metrics.accuracy_score(Y_test, test_data_pre)
print("Test Data Accuracy :", test_acc_score)

test_data_pre = nn.predict_proba(X_test)
test_top_k_acc_score = top_n_accuracy(test_data_pre, Y_test.values, 3)
print("Test Data Top 3 Accuracy :", test_top_k_acc_score)

get_classes_prob(test_data_pre, Y_test.values, 3)

'''
Train Data Accuracy : 0.8191531460229521
Train Data Top 3 Accuracy : 97.90265136525524
Test Data Accuracy : 0.5297783933518005
Test Data Top 3 Accuracy : 87.18836565096953
i = 0 , total = 456.0 , correct = 456.0 , prob = 100.0
i = 1 , total = 524.0 , correct = 494.0 , prob = 94.27480916030534
i = 2 , total = 377.0 , correct = 362.0 , prob = 96.02122015915118
i = 3 , total = 544.0 , correct = 499.0 , prob = 91.72794117647058
i = 4 , total = 449.0 , correct = 362.0 , prob = 80.62360801781738
i = 5 , total = 691.0 , correct = 603.0 , prob = 87.26483357452966
i = 6 , total = 468.0 , correct = 359.0 , prob = 76.70940170940172
i = 7 , total = 171.0 , correct = 97.0 , prob = 56.72514619883041
i = 8 , total = 8.0 , correct = 2.0 , prob = 25.0
i = 9 , total = 152.0 , correct = 94.0 , prob = 61.8421052631579
i = 10 , total = 492.0 , correct = 449.0 , prob = 91.26016260162602
'''