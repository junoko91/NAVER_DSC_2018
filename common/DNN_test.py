import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import datetime

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

size_l1 =4096
size_l2 = size_l1/2
size_l3 =size_l2/2
size_l4 =size_l3/2

rate = 0.97
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

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))

accuracy0 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(hypothesis, tf.argmax(Y_one_hot, 1), k=1), tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(hypothesis, tf.argmax(Y_one_hot, 1), k=2), tf.float32))
accuracy3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(hypothesis, tf.argmax(Y_one_hot, 1), k=3), tf.float32))

is_top1 = tf.equal(tf.nn.top_k(hypothesis, k=1)[1][:, 0], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_top2 = tf.equal(tf.nn.top_k(hypothesis, k=2)[1][:, 1], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_top3 = tf.equal(tf.nn.top_k(hypothesis, k=3)[1][:, 2], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_top4 = tf.equal(tf.nn.top_k(hypothesis, k=3)[1][:, 2], tf.cast(tf.argmax(Y_one_hot, 1), tf.int32))
is_in_top1 = is_top1
is_in_top2 = tf.logical_or(is_in_top1, is_top2)
is_in_top3 = tf.logical_or(is_in_top2, is_top3)
is_in_top4 = tf.logical_or(is_in_top3, is_top4)

accuracy11 = tf.reduce_mean(tf.cast(is_in_top1, tf.float32))
accuracy22 = tf.reduce_mean(tf.cast(is_in_top2, tf.float32))
accuracy33 = tf.reduce_mean(tf.cast(is_in_top3, tf.float32))
accuracy44 = tf.reduce_mean(tf.cast(is_in_top4, tf.float32))

print("train start")

epoch = 200000
batch_size = 128
mok = int(len(X_train) / batch_size)

position_dict = {0:"GK",1:"LB",2:"CB",3:"RB",4:"LM",5:"CM",6:"RM",7:"LW",8:"RW",9:"CF",10:"ST"}

saver = tf.train.Saver()

index = 1

def train_start():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch * mok):
            if index <= mok:
                x_train_batch = X_train[(index - 1) * batch_size: index * batch_size, :]
                y_train_batch = y_train[(index - 1) * batch_size: index * batch_size, :]
            else:
                x_train_batch = X_train[(index - 1) * batch_size: -1, :]
                y_train_batch = y_train[(index - 1) * batch_size: -1, :]
                index = 0

            index += 1

            sess.run(optimizer, feed_dict={X: x_train_batch, Y: y_train_batch})

            if i % mok == 0:
                loss, acc = sess.run([cost, accuracy11], feed_dict={X: X_train, Y: y_train})
                print("<", i, ' :', 'loss : ', loss, ' acc : ', acc * 100, ">")
                if int(acc*1000) >= 983:
                    loss, acc,hypo = sess.run([cost, accuracy44,hypothesis], feed_dict={X: X_test, Y: y_test})
                    print(i, ' :', 'loss : ', loss, ' acc : ', acc * 100)
                    top3 = tf.nn.top_k(hypo, k=3)[1].eval()
                    for ii in range(3):
                        print(ii," : ",position_dict[top3[0][ii]]," ")
                    print("")
                    saver.save(sess,"./model/"+str(i)+"model.ckpt")
                    break
                else:
                    loss, acc = sess.run([cost, accuracy44], feed_dict={X: X_test, Y: y_test})
                    print(i, ' :', 'loss : ', loss, ' acc : ', acc * 100)

def predict(x,model_name):
    with tf.Session() as sess:
        saver.restore(sess, model_name)
        sess.run(tf.global_variables_initializer())

        loss, acc = sess.run([cost, accuracy44], feed_dict={X: X_test, Y: y_test})
        print( 'loss : ', loss, ' acc : ', acc * 100)