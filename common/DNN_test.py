import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

data = pd.read_csv("football_players_test1.csv")

data = shuffle(data)

data.info()

X = data.drop("Club_Position",axis=1)
X = X.drop("Club_Position1",axis=1)
y = data["Club_Position1"]

X.head()
y.head()

# pca = PCA(n_components=15)
# X = pca.fit_transform(X)

# mdict = dict(enumerate(y.unique()))
# y.replace(to_replace=mdict.values(),value=mdict.keys(),inplace=True)

X=StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# sm = SMOTE()
# X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

print(np.shape(X_train))
print(np.shape(y_train))
print(y_train.ravel())

y_test = np.reshape(np.asarray(y_test),[len(y_test),1])
y_train = np.reshape(np.asarray(y_train.ravel()),[len(y_train),1])

print(y.unique())

nbclass = len(y.unique())
nbfeature = np.shape(X_test)[1]

X = tf.placeholder(tf.float32,[None,nbfeature])
Y = tf.placeholder(tf.int32,[None,1])
Y_one_hot = tf.one_hot(Y,nbclass)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nbclass])

keep_prob = tf.placeholder(tf.float32)

size_l1 =1024
size_l2 = size_l1/2
size_l3 =size_l2/2
size_l4 =size_l3/2

rate = 0.3
l1 = tf.layers.dense(X,size_l1,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.xavier_initializer())
l1 = tf.layers.dropout(l1,rate=rate)
l2 = tf.layers.dense(l1,size_l2,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.xavier_initializer())
l2 = tf.layers.dropout(l2,rate=rate)
l3 = tf.layers.dense(l2,size_l3,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.xavier_initializer())
l3 = tf.layers.dropout(l3,rate=rate)
l4 = tf.layers.dense(l3,size_l4,activation=tf.nn.relu,bias_initializer=tf.contrib.layers.xavier_initializer())
l4 = tf.layers.dropout(l4,rate=rate)
l5 = tf.layers.dense(l1,nbclass,activation=None)

hypothesis = tf.nn.softmax(l5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l5,labels=Y_one_hot))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

print("train start")


sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

epoch = 200000
batch_size = 128
mok = int(len(X_train) / batch_size)

index = 1
for i in range(epoch*mok):
    if index <= mok:
        x_train_batch = X_train[(index-1)*batch_size : index*batch_size , :]
        y_train_batch = y_train[(index-1)*batch_size : index*batch_size , :]
    else:
        x_train_batch = X_train[(index - 1) * batch_size: -1, :]
        y_train_batch = y_train[(index - 1) * batch_size: -1, :]
        index = 0

    index += 1

    sess.run(optimizer,feed_dict={X: x_train_batch,Y:y_train_batch})

    if i % mok == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={X: X_train, Y: y_train})
        print("<",i, ' :', 'loss : ', loss, 'acc : ', acc * 100,">")
        loss,acc,hypo = sess.run([cost,accuracy,hypothesis],feed_dict={X: X_test,Y:y_test})
        print(i, ' :', 'loss : ', loss, 'acc : ', acc*100,hypo[0:1]*100)
